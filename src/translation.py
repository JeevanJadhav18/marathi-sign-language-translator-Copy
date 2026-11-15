# Marathi Sign Language Translation Dictionary

SIGN_TRANSLATIONS = {
    "hello": "नमस्कार",
    "thank_you": "धन्यवाद",
    "sorry": "माफ करा",
    "yes": "हो",
    "no": "नाही"
}


def translate_to_marathi(predicted_sign):
    return SIGN_TRANSLATIONS.get(
        predicted_sign,
        "चिन्ह ओळखले नाही"
    )
