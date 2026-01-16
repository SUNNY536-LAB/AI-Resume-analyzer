import re

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    sample_text = "Experienced Python Developer with Flask knowledge."
    print(clean_text(sample_text))
