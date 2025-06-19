import tenseal as ts
import numpy as np
from extract_features import extract_feature
from encrypted_classifier import classify_encrypted

LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

def classify_encrypted_image(img_path):
    print(f" معالجة الصورة: {img_path}")

    # 1. استخراج الميزات من الصورة
    features = extract_feature(img_path)

    # 2. إنشاء سياق CKKS وتشفير الميزات
    context = create_context()
    enc_vector = ts.ckks_vector(context, features)

    # 3. تطبيق التصنيف المشفر (ثم فك التشفير داخليًا)
    prediction_vector = classify_encrypted(enc_vector)

    # 4. تحليل النتائج
    detected = []
    for i, score in enumerate(prediction_vector):
        print(f" {LABELS[i]} → القيمة: {score:.3f}")
        if score > 0:
            detected.append(LABELS[i])

    # 5. طباعة النتيجة النهائية
    if detected:
        print(" الأمراض المكتشفة:", " | ".join(detected))
    else:
        print(" لا توجد أمراض ظاهرة")

#  اختبر الآن على صورة فعلية
classify_encrypted_image("data/Full_Training_Data/2_left.jpg")
