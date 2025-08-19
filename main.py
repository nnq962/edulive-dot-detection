from circle_detector import CircleDetector

model = CircleDetector()

model.predict(
    source="test_images/IMG_0869.MOV",
    mode="contour",
    show=True,
    verbose=True,
    save=False
)