import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor
from ultralytics import YOLO

sam2_checkpoint = "MODEL_CHECKPOINT"
model_cfg = "MODEL_CONFIG"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

cap = cv2.VideoCapture(1)
if_init = False
click_point = None
reference_background = None
cloak_active = False

def mouse_callback(event, x, y, flags, param):
    global click_point, if_init
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = [x, y]
        if_init = False  # Reset to add new prompt
        print(f"Clicked at: ({x}, {y}) - Will segment object here for cloaking!")

cv2.namedWindow('Invisibility Cloak', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Invisibility Cloak', mouse_callback)

print("Instructions:")
print("1. Press 'r' to capture reference background (make sure the area is clear!)")
print("2. Click on the object you want to make invisible")
print("3. Press 'c' to toggle cloak effect on/off")
print("4. Press 'q' to quit")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Capture reference background
        if key == ord('r'):
            reference_background = frame.copy()
            print("Reference background captured! Now click on object to cloak.")
        
        # Toggle cloak effect
        elif key == ord('c'):
            cloak_active = not cloak_active
            print(f"Cloak effect: {'ON' if cloak_active else 'OFF'}")
        
        # Quit
        elif key == ord('q'):
            break

        # Show instructions on frame
        if reference_background is None:
            pass
            # cv2.putText(frame, "Press 'r' to capture background", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            status = "CLOAKING" if cloak_active else "READY"
            
            # cv2.putText(frame, f"Click object to cloak - {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, "Press 'c' to toggle cloak", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Initialize segmentation when clicked
        if click_point is not None and not if_init and reference_background is not None:
            predictor.load_first_frame(frame)
            if_init = True

            frame_idx_out, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0,
                obj_id=1,
                points=[click_point],
                labels=[1]
            )
            print(f"Segmenting object at {click_point} for invisibility cloak!")
            
        elif if_init:
            out_obj_ids, out_mask_logits = predictor.track(frame)

        # Apply invisibility cloak effect
        if if_init and out_mask_logits is not None and len(out_mask_logits) > 0 and reference_background is not None:
            masks = torch.sigmoid(out_mask_logits)
            masks = (masks > 0.5).float()

            mask = masks[0, 0].cpu().numpy()
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            if cloak_active:
                # Create invisibility effect - replace segmented area with background
                mask_3d = np.stack([mask, mask, mask], axis=2)
                frame = np.where(mask_3d > 0.5, reference_background, frame).astype(np.uint8)
            else:
                # Show segmentation overlay for debugging
                overlay = frame.copy()
                overlay[mask > 0.5] = [0, 255, 0]
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Draw click point
            if click_point and not cloak_active:
                cv2.circle(frame, tuple(click_point), 8, (255, 0, 0), -1)

        cv2.imshow('Invisibility Cloak', frame)

cap.release()
cv2.destroyAllWindows()
