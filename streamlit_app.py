
import streamlit as st
from io import BytesIO
import os
import time
import cv2
import mediapipe as mp
import numpy as np


def is_allowed_file(filename):
    # List of allowed file extensions
    allowed_extensions = ['.mp4']

    # Get the file extension
    _, ext = os.path.splitext(filename)
    return ext.lower() in allowed_extensions


def housekeeping(file):
    os.remove(f'./{file}')
    time.sleep(5)
    st.write('File successfully downloaded')


def vid_download(filename):
    if st.button("Click Me!"):
        st.write(filename)
    try:
        with open(f'{filename}', 'rb') as fh:
            buf = BytesIO(fh.read())
        fh.close()

        if st.download_button(label='Download video', data=buf, file_name=filename, mime='video/mp4'):
            housekeeping(filename)

    except Exception as down_err:
        st.write(f'Error is: {down_err}')


def blur_face_mediapipe(input_video, output_video, blur_amount=81):
    """
    Blurs faces in a video using MediaPipe for face detection and gaussian blur.

    Args:
        input_video: Path to the input video file.
        output_video: Path to save the output video file.
        blur_amount: Integer controlling the blur amount (higher = more blur).
    """

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'x264', 'avc1'
    st.write(' test: ', output_video)
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB, needed by mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    x, y, w, h = bbox
                    if w > 0 and h > 0:  # Avoid potential errors with zero-area boxes

                        face_roi = frame[y:y + h, x:x + w]
                        if face_roi.size > 0:  # Make sure the ROI is valid
                            blurred_face = cv2.GaussianBlur(face_roi, (blur_amount, blur_amount), 0)
                            frame[y:y + h, x:x + w] = blurred_face

            out.write(frame)
    except Exception as e:
        print(f"Error processing the video: {e}")
    finally:
        cap.release()
        out.release()
        face_detection.close()
        cv2.destroyAllWindows()


def vid_upload():
    st.sidebar.title("Sidebar")
    st.sidebar.write("This is the sidebar.")

    st.sidebar.write("value", st.sidebar.slider("Select a value", 0, 100, 50))

    src_file = st.file_uploader("Upload a file")
    if src_file:

        if is_allowed_file(src_file.name):
            st.write("File uploaded successfully!")
            file, ext = os.path.splitext(src_file.name)
            # input_video_path = './' + src_file.name
            st.write('file: ', file)
            input_video_path = src_file.name
            output_video_path = './' + file + '_mp.mp4'
            st.write(output_video_path)

            blur_face_mediapipe(input_video_path, output_video_path)

            vid_download(output_video_path)
        else:
            st.write(f'Invalid file type: {src_file.name}')


def set_config():
    # setting page icon
    st.set_page_config(page_title='Download video',
                       page_icon='timer_clock', initial_sidebar_state='auto')

    # hide hamburger menu and footer logo
    hide_st_style = """
              <style>
              #MainMenu {visibility: hidden;}
              footer {visibility: hidden;}
              </style>
              """
    st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == '__main__':
# calling the page config
    set_config()

    st.title("My Streamlit App in Colab")
    st.write("Hello from Google Colab!")

# execute the app
    vid_upload()


