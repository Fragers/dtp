import cv2


class MyAnnotator:
    @staticmethod
    def put_text(frame, text: str, position: tuple[int, int], scale=1, color=(0, 0, 255), thickness=2):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    @staticmethod
    def rectangle(frame, start: tuple[int, int], end: tuple[int, int], color=(0, 0, 255), thickness=2):
        cv2.rectangle(frame, start, end, color, thickness)
