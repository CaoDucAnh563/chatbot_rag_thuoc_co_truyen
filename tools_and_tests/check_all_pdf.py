import os
import fitz

folder = "data"

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        path = os.path.join(folder, file)
        try:
            doc = fitz.open(path)
            total_chars = sum(len(page.get_text()) for page in doc)
            
            print(
                f"{file:50} | "
                f"{len(doc):4} trang | "
                f"{total_chars:10} ký tự"
            )
            doc.close()
        except Exception as e:
            print(file, e)