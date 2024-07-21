import argparse
import cv2 as cv
import numpy as np
import os


def visualize(image, faces, thickness=2):
    """
    Visualize detected faces on the image.

    Parameters:
    - image: The image on which to draw rectangles and circles.
    - faces: Detected faces with coordinates.
    - thickness: Thickness of the lines used to draw rectangles and circles.
    """
    for idx, face in enumerate(faces[1]):
        coords = face[:-1].astype(np.int32)
        cv.rectangle(
            image,
            (coords[0], coords[1]),
            (coords[0] + coords[2], coords[1] + coords[3]),
            (0, 255, 0),
            thickness
        )
        for i in range(4, 14, 2):
            cv.circle(image, (coords[i], coords[i + 1]), 2, (0, 255, 0), thickness)


def detect_and_match(query_image, aadhaar_image, face_detector, face_recognizer,
                     score_threshold, nms_threshold, top_k, cosine_similarity_threshold,
                     l2_similarity_threshold):
    """
    Detect and match faces between a query image and an Aadhaar image.

    Parameters:
    - query_image: The query image to compare.
    - aadhaar_image: The Aadhaar image to compare with.
    - face_detector: The face detection model.
    - face_recognizer: The face recognition model.
    - score_threshold: Minimum confidence score for face detection.
    - nms_threshold: Non-Maximum Suppression threshold.
    - top_k: Number of top detected faces to process.
    - cosine_similarity_threshold: Threshold for cosine similarity to consider faces as the same identity.
    - l2_similarity_threshold: Threshold for L2 distance to consider faces as the same identity.

    Returns:
    - is_same_identity: Boolean indicating if the faces are considered the same identity.
    - message: Message with cosine similarity and L2 distance percentages.
    """
    face_detector.setInputSize((query_image.shape[1], query_image.shape[0]))
    face_in_query = face_detector.detect(query_image)
    if face_in_query[1] is None:
        return False, "No face detected in query image"

    face_detector.setInputSize((aadhaar_image.shape[1], aadhaar_image.shape[0]))
    face_in_aadhaar = face_detector.detect(aadhaar_image)
    if face_in_aadhaar[1] is None:
        return False, "No face detected in Aadhaar image"

    face1_align = face_recognizer.alignCrop(query_image, face_in_query[1][0])
    face2_align = face_recognizer.alignCrop(aadhaar_image, face_in_aadhaar[1][0])

    face1_feature = face_recognizer.feature(face1_align)
    face2_feature = face_recognizer.feature(face2_align)

    cosine_score = face_recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
    l2_score = face_recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)

    # Convert scores to percentages
    cosine_score_percentage = cosine_score * 100
    l2_score_percentage = (1 - (l2_score / l2_similarity_threshold)) * 100  # Inverted distance percentage

    # Determine if faces are the same identity
    is_same_identity = (
        cosine_score_percentage >= (cosine_similarity_threshold * 100)
        and l2_score_percentage >= (100 - (l2_similarity_threshold * 100))
    )

    return (
        is_same_identity,
        f"Cosine Similarity (for similar it should be > {cosine_similarity_threshold * 100:.1f}%): {cosine_score_percentage:.2f}%\n"
        f"NormL2 Distance (for similar it should be > 0%): {l2_score_percentage:.2f}%"
    )


def main():
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--reference_directory", required=True, help="Path to the directory containing Aadhaar images")
    ap.add_argument("-q", "--query_image", required=True, help="Path to the query image")
    args = vars(ap.parse_args())

    aadhaar_dir = args["reference_directory"]
    query_image_path = args["query_image"]

    # Read query image
    query_image = cv.imread(query_image_path)

    # Define detection and recognition parameters
    score_threshold = 0.9
    nms_threshold = 0.3
    top_k = 5000
    cosine_similarity_threshold = 0.363
    l2_similarity_threshold = 1.128

    # Load models
    face_detector = cv.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx", "", (0, 0),
        score_threshold, nms_threshold, top_k
    )
    face_recognizer = cv.FaceRecognizerSF.create(
        "face_recognition_sface_2021dec.onnx", ""
    )

    total_aadhaar_images = 0
    successful_matches = 0
    matched_aadhaar_images = []

    # Process each Aadhaar image
    for aadhaar_image_name in os.listdir(aadhaar_dir):
        aadhaar_image_path = os.path.join(aadhaar_dir, aadhaar_image_name)
        aadhaar_image = cv.imread(aadhaar_image_path)
        if aadhaar_image is None:
            continue

        total_aadhaar_images += 1
        is_same_identity, message = detect_and_match(
            query_image, aadhaar_image, face_detector, face_recognizer,
            score_threshold, nms_threshold, top_k, cosine_similarity_threshold,
            l2_similarity_threshold
        )

        print("------------------------------------------------------------------------------------------------------------")
        print(f"Processing {aadhaar_image_name}:")
        print(f"   - Cosine Similarity (for similar it should be > {cosine_similarity_threshold * 100:.1f}%): {message.split('\n')[0].split(': ')[1]}")
        print(f"   - NormL2 Distance (for similar it should be > 0%): {message.split('\n')[1].split(': ')[1]}")

        if is_same_identity:
            successful_matches += 1
            matched_aadhaar_images.append(aadhaar_image_name)

    # Calculate accuracy
    if total_aadhaar_images > 0:
        accuracy_percentage = (successful_matches / total_aadhaar_images) * 100
        print("------------------------------------------------------------------------------------------------------------")
        print(f"Accuracy: {accuracy_percentage:.2f}% ({successful_matches}/{total_aadhaar_images})")
        print("------------------------------------------------------------------------------------------------------------")
        if matched_aadhaar_images:
            print("Matched Aadhaar images:")
            for img_name in matched_aadhaar_images:
                print(f"- {img_name}")
    else:
        print("No valid Aadhaar images processed")
    print("------------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
