import os
import cv2

def take_screenshot_from_video(video_path, screenshot_path):
    video_capture = cv2.VideoCapture(video_path)
    
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, fps)  
    
    success, frame = video_capture.read()
    
    if success:
        cv2.imwrite(screenshot_path, frame)
    else:
        print(f"error while reading video: {video_path}")
    
    video_capture.release()

def process_videos(real_folder, synthetic_folder, output_folder):
    for real_file in os.listdir(real_folder):
        if real_file.endswith(".mp4"):
            real_name = os.path.splitext(real_file)[0]  # Example: 'id0_0000'
            real_id, real_variant = real_name.split('_')  # Example: 'id0', '0000'
            real_video_path = os.path.join(real_folder, real_file)
            
            # create main folder for this person (z.B. 'id0')
            person_folder = os.path.join(output_folder, real_id)
            os.makedirs(person_folder, exist_ok=True)
            
            # create subfolder for each variant of this person (z.B. '0000', '0001')
            variant_folder = os.path.join(person_folder, real_variant)
            os.makedirs(variant_folder, exist_ok=True)
            
            # screenshot of video
            real_screenshot_path = os.path.join(variant_folder, real_name + ".jpg")
            take_screenshot_from_video(real_video_path, real_screenshot_path)
            print(f"screenshot for real file {real_file} created.")
            
            # search of synthetic files of real person
            for synthetic_file in os.listdir(synthetic_folder):
                synthetic_name = os.path.splitext(synthetic_file)[0]  # example: 'id0_id1_0000'
                synthetic_id, synthetic_variant, synthetic_real_variant = synthetic_name.split('_')  # example: 'id0', 'id1', '0000'
                
                # get only fake data files of same ID and variant
                if synthetic_id == real_id and synthetic_real_variant == real_variant:
                    synthetic_video_path = os.path.join(synthetic_folder, synthetic_file)
                    
                    # screenshot of fake video
                    synthetic_screenshot_path = os.path.join(variant_folder, synthetic_name + ".jpg")
                    take_screenshot_from_video(synthetic_video_path, synthetic_screenshot_path)
                    print(f"screenshot for synthetic file {synthetic_file} created.")

if __name__ == "__main__":
    real_folder = input("Enter path of folder 'Celeb-real': ")
    synthetic_folder = input("Enter path of folder 'Celeb-synthesis': ")
    output_folder = input("Enter path of output folder: ")
    
    process_videos(real_folder, synthetic_folder, output_folder)