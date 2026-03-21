def ocr_image_file(image_path: str, engine: callable, **kwargs) -> dict:
    try:
        result = engine(image_path, **kwargs)
        return result

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return {"text": "", "confidence": 0.0}