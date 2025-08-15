from circle_detector import CircleDetector

model = CircleDetector()

model.predict(
    source=0,
    mode="contour",
    show=True,
    verbose=True,
    save=False
)