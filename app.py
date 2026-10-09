import os
import re
import joblib
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(BASE_DIR, "model_bundle.joblib"))
model = bundle["model"]
mlb = bundle["mlb"]
SYMPTOMS = bundle["symptoms"]
DESCRIPTIONS = bundle["descriptions"]
PRECAUTIONS = bundle["precautions"]
SEVERITY_WEIGHTS = bundle["severity_weights"]

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
app = Flask(__name__)

def norm(text):
    text = str(text or "").strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)

def map_free_text_to_symptoms(text):
    """Conservative exact/phrase matching against the dataset vocabulary."""
    s = norm(text)
    found = set()
    # Longest-first phrase matching reduces partial overlaps.
    for symptom in sorted(SYMPTOMS, key=len, reverse=True):
        if symptom and re.search(r"(?<![a-z])" + re.escape(symptom) + r"(?![a-z])", s):
            found.add(symptom)
    return found

def specialist_for(symptoms, body_part):
    s = " ".join(symptoms).lower()
    if body_part == "Skin" or any(x in s for x in ["skin rash", "itching", "pus filled pimples", "scurring"]):
        return "Dermatologist"
    if body_part in ("Knee / joints", "Back / neck") or any(x in s for x in [
        "knee pain", "joint pain", "back pain", "neck pain", "muscle weakness",
        "movement stiffness", "swelling joints"
    ]):
        return "Orthopedist"
    if body_part == "Dental" or any(x in s for x in ["toothache", "ulcers on tongue"]):
        return "Dentist"
    if body_part == "Eye":
        return "Ophthalmologist"
    if any(x in s for x in ["depression", "anxiety", "mood swings"]):
        return "Mental Health Professional"
    return "General Physician"

def safety_assessment(symptoms, free_text, severity):
    combined = (free_text + " " + " ".join(symptoms)).lower()
    # Keyword screening is incomplete and cannot rule out an emergency.
    danger_terms = [
        "chest pain", "difficulty breathing", "shortness of breath",
        "cannot breathe", "can't breathe", "unconscious", "fainting",
        "face drooping", "one sided weakness", "sudden weakness",
        "heavy bleeding", "uncontrolled bleeding", "seizure", "overdose"
    ]
    matches = [term for term in danger_terms if term in combined]
    if matches:
        return True, (
            "Possible warning sign detected. Do not wait for an AI result. "
            "Seek urgent medical help. In India, call 112 for a life-threatening emergency."
        )
    if severity == "Severe":
        return True, (
            "You selected severe symptoms. Seek prompt medical assessment and do not "
            "start exercises to treat the symptom before assessment."
        )
    return False, (
        "This checker cannot rule out an emergency. If symptoms are sudden, severe, "
        "or rapidly worsening, seek urgent medical care."
    )

def activity_guidance(symptoms, body_part, severity, free_text):
    combined = (free_text + " " + " ".join(symptoms)).lower()
    if severity == "Severe":
        return ("Do not start an exercise programme. Avoid movements that worsen symptoms "
                "and seek prompt medical assessment.")
    if any(x in combined for x in [
        "knee pain", "knee swelling", "swelling joints", "joint pain",
        "ligament", "sprain", "fracture", "joint stiffness"
    ]) or body_part == "Knee / joints":
        return ("Avoid running, jumping, deep squats and forced joint movements if there is "
                "pain, swelling or instability. Do not exercise through pain. Ask an "
                "orthopedist or physiotherapist to assess the problem before starting "
                "specific rehabilitation exercises.")
    if any(x in combined for x in [
        "chest pain", "breathlessness", "difficulty breathing", "dizziness", "fainting"
    ]):
        return ("Do not exercise while these symptoms are unexplained. Seek medical assessment; "
                "sudden or severe symptoms may be an emergency.")
    if body_part == "Back / neck":
        return ("Avoid heavy lifting and movements that worsen pain. Gentle comfortable movement "
                "may be appropriate in some uncomplicated cases, but persistent, severe or "
                "worsening pain should be assessed.")
    if body_part == "Skin":
        return ("Exercise does not treat most skin conditions. Avoid suspected irritants and seek "
                "care for spreading, painful, infected-looking or fever-associated rashes.")
    return ("The dataset does not contain validated exercise prescriptions. There is not enough "
            "information to choose a specific exercise safely. Avoid activities that worsen "
            "symptoms; ask a clinician if symptoms persist.")

def ayurveda_guidance(disease, symptoms, severity, urgent):
    """Evidence-aware complementary-care information; never a cure prescription."""
    name = norm(disease)
    if urgent or severity == "Severe":
        return {
            "evidence_label": "Do not delay medical care",
            "role": "Ayurveda must not be used to manage a possible emergency or severe symptoms at home. Seek urgent medical assessment first.",
            "safer_support": "Follow emergency/clinician advice. Do not start herbs, oils, detoxes, fasting or exercise as treatment.",
            "avoid": "Do not replace prescribed treatment or delay urgent care with Ayurvedic remedies.",
        }

    if "osteoarth" in name:
        return {
            "evidence_label": "Limited evidence for symptom relief; not a cure",
            "role": "A few small or imperfect studies suggest some Ayurvedic preparations may help osteoarthritis pain or function, but evidence is not strong enough to say they cure arthritis or repair cartilage.",
            "safer_support": "Discuss any Ayurveda plan with your doctor. Ask about a physiotherapist-approved movement plan, weight management if relevant, sleep and pacing activities. Gentle movement should not worsen pain.",
            "avoid": "Do not self-prescribe herbal/mineral products or stop pain medicines. Some preparations may contain toxic metals or interact with medicines.",
        }
    if "arthritis" in name:
        return {
            "evidence_label": "Insufficient evidence to claim disease control or cure",
            "role": "Evidence for Ayurveda in arthritis is limited. It must not replace diagnosis or disease-modifying treatment, particularly for inflammatory arthritis.",
            "safer_support": "Discuss complementary care with your treating clinician and use only activity or physiotherapy guidance appropriate for your diagnosis.",
            "avoid": "Do not stop prescribed medicines or use herbal/mineral preparations without a clinician/pharmacist review.",
        }
    if "diabetes" in name or "hypoglycemia" in name:
        return {
            "evidence_label": "Limited/inconclusive evidence; not a substitute for treatment",
            "role": "Some small studies have explored Ayurvedic formulations for type 2 diabetes, but evidence is insufficient to recommend them as a cure or replacement for standard care. Hypoglycaemia can be urgent.",
            "safer_support": "Use clinician-guided nutrition, regular monitoring, sleep and activity recommendations. Ask your clinician before any supplement because blood sugar may change.",
            "avoid": "Never stop insulin or diabetes medicines on your own. Avoid unverified products and seek urgent help for severe low blood sugar or confusion.",
        }
    if any(term in name for term in ["hepatitis", "jaundice", "cholestasis", "liver"]):
        return {
            "evidence_label": "No proven Ayurvedic cure; medical assessment is important",
            "role": "There is not reliable evidence that Ayurveda cures hepatitis, jaundice or liver disease. Some herbal products can injure the liver.",
            "safer_support": "Get medical assessment and follow liver-specific advice. Tell the clinician about all herbs, supplements and traditional medicines you use.",
            "avoid": "Avoid self-treatment, detox products and unverified herbal/mineral remedies; do not delay tests or prescribed treatment.",
        }
    if any(term in name for term in ["malaria", "dengue", "typhoid", "tuberculosis", "pneumonia", "gastroenteritis", "urinary tract", "infection", "impetigo", "chicken pox", "aids", "heart attack", "paralysis"]):
        return {
            "evidence_label": "Not a replacement for proven medical treatment",
            "role": "No Ayurvedic remedy should be presented as a proven cure for this possible condition. Timely medical assessment and condition-specific treatment can be important.",
            "safer_support": "Use only supportive steps recommended by a clinician, such as appropriate fluids or rest when suitable for your condition.",
            "avoid": "Do not delay medical testing, antibiotics/antivirals or emergency treatment when indicated. Avoid self-prescribed herbs and mineral preparations.",
        }
    return {
        "evidence_label": "Insufficient evidence for a proven cure",
        "role": "For this possible condition, reliable evidence does not establish Ayurveda as a cure. It may be discussed as complementary care only after the diagnosis and safety needs are understood.",
        "safer_support": "General supportive habits may include a balanced diet, adequate sleep and comfortable, appropriate activity. Choose them according to your clinician's advice.",
        "avoid": "Do not use herbs, oils, detoxes or mineral preparations as a substitute for medical assessment or prescribed treatment. Check interactions with a doctor or pharmacist.",
    }


def diet_guidance(disease, urgent=False, severity="Mild"):
    """General educational food guidance; not a personalized diet prescription."""
    name = norm(disease)
    if urgent or severity == "Severe":
        return {
            "title": "Food guidance is not the priority right now",
            "eat": "Seek urgent or prompt medical assessment first. Follow the clinician's advice about food and fluids.",
            "limit": "Do not delay care to try a diet, detox, fasting plan or home remedy.",
            "caution": "If the person is unconscious, confused, choking, or unable to swallow safely, do not give food or drink by mouth. Call emergency services when needed."
        }
    if "diabetes" in name:
        return {
            "title": "If diabetes is confirmed",
            "eat": "Choose regular, balanced meals with non-starchy vegetables, pulses/beans, whole grains in suitable portions, and unsweetened drinks. Pair carbohydrate foods with protein/fibre when practical; follow your clinician's glucose-monitoring and meal plan.",
            "limit": "Limit sugary drinks, sweets and frequent large portions of refined snacks. Do not assume all rice, roti or fruit must be eliminated; portion and personal glucose response matter.",
            "caution": "Do not change insulin or diabetes medicines based on this website. Low blood sugar needs a personal treatment plan; severe confusion or unconsciousness is an emergency."
        }
    if "hypoglycemia" in name:
        return {
            "title": "Possible low blood sugar needs careful handling",
            "eat": "If the person is awake and can swallow safely, follow their established hypoglycaemia plan for fast-acting carbohydrate and recheck glucose as instructed.",
            "limit": "Do not delay the person's prescribed low-glucose treatment to try a regular meal or herbal remedy.",
            "caution": "If unconscious, having a seizure, or unable to swallow safely, do not give food or drink by mouth; call emergency services and follow their emergency plan."
        }
    if "hypertension" in name:
        return {
            "title": "If high blood pressure is confirmed",
            "eat": "A balanced pattern with vegetables, fruit, beans/pulses, whole grains and unsalted nuts can help. Prefer home-cooked foods and check labels for sodium.",
            "limit": "Reduce excess salt, salty packaged snacks, instant noodles, pickles/papad and highly processed foods. WHO recommends adults consume under 5 g salt per day from all sources.",
            "caution": "Do not use potassium-based salt substitutes without medical advice, especially with kidney disease or medicines that raise potassium. Diet does not replace prescribed BP treatment."
        }
    if "gerd" in name or "acid reflux" in name:
        return {
            "title": "If reflux/GERD is confirmed",
            "eat": "Choose foods you tolerate and consider smaller meals if large meals trigger symptoms.",
            "limit": "Track personal triggers; common ones include spicy or high-fat foods, coffee/caffeine, chocolate, mint, citrus/tomato and alcohol, but not everyone needs to avoid all of them. Avoid lying down for about 3 hours after eating if night reflux occurs.",
            "caution": "Seek medical advice for trouble swallowing, vomiting blood, black stools, unexplained weight loss or persistent symptoms."
        }
    if "gastroenteritis" in name:
        return {
            "title": "If gastroenteritis is confirmed",
            "eat": "Take frequent small sips of fluids. Oral rehydration solution (ORS) can help replace fluid and salts lost through diarrhoea/vomiting; eat simple foods as tolerated.",
            "limit": "Avoid alcohol and consider avoiding very fatty/spicy foods. Fruit juice and fizzy drinks may worsen diarrhoea.",
            "caution": "Seek care for dehydration, blood in stool/vomit, severe pain, persistent vomiting, high fever, or symptoms in a young child, older adult or medically vulnerable person. Follow clinician advice if fluid-restricted."
        }
    if any(x in name for x in ["hepatitis", "jaundice", "cholestasis", "liver"]):
        return {
            "title": "Liver-related symptoms need clinical assessment",
            "eat": "Eat regular balanced meals as tolerated; a clinician may tailor nutrition to the specific liver condition.",
            "limit": "Avoid alcohol until a clinician confirms it is safe. Avoid detoxes and unverified herbs/supplements because some can injure the liver.",
            "caution": "Do not self-treat jaundice or hepatitis with a diet. Seek prompt medical assessment and disclose all supplements and medicines."
        }
    if "peptic ulcer" in name:
        return {
            "title": "If peptic ulcer disease is confirmed",
            "eat": "There is no single proven ulcer-healing diet; choose balanced foods that you tolerate and keep regular meals if that feels better.",
            "limit": "Limit foods or drinks that personally worsen symptoms, including alcohol if it triggers discomfort. Do not eliminate broad food groups without a reason.",
            "caution": "Ulcers often need medical treatment and testing for causes such as H. pylori. Ask a clinician before using anti-inflammatory pain medicines; urgent care is needed for vomiting blood or black stools."
        }
    if "dimorphic hemmorhoids" in name or "piles" in name or "hemorrhoid" in name:
        return {
            "title": "If piles/haemorrhoids are confirmed",
            "eat": "Increase fibre gradually with vegetables, fruit, beans/pulses and whole grains. Drink enough water unless a clinician has restricted fluids.",
            "limit": "Avoid straining and long periods on the toilet. Do not rely on food changes alone if there is persistent pain, a lump or bleeding.",
            "caution": "Rectal bleeding or persistent symptoms need medical assessment; do not assume every symptom is piles."
        }
    if "allergy" in name:
        return {
            "title": "Avoid only a known or clinician-confirmed trigger",
            "eat": "Maintain a balanced diet with foods you know you tolerate.",
            "limit": "Do not remove multiple food groups based only on an AI match. Avoid a food only when there is a credible reaction history or professional advice.",
            "caution": "Food allergy with breathing difficulty, throat/tongue swelling, faintness or rapid worsening is an emergency."
        }
    if "acne" in name:
        return {
            "title": "Nutrition may support general skin health, but is not a cure",
            "eat": "Choose a balanced diet with vegetables, fruit, pulses and whole grains.",
            "limit": "Evidence linking individual foods to acne is mixed. If you suspect a personal trigger, track it and discuss it with a clinician rather than starting a restrictive diet.",
            "caution": "Do not stop prescribed acne treatment or take high-dose supplements based on an online prediction."
        }
    if any(x in name for x in ["malaria", "dengue", "typhoid", "tuberculosis", "pneumonia", "urinary tract", "infection", "chicken pox"]):
        return {
            "title": "Supportive nutrition only; medical treatment matters",
            "eat": "Take fluids and balanced foods as tolerated, unless a clinician has told you to restrict fluids or follow a special diet.",
            "limit": "Avoid alcohol and unverified detox/herbal products. Avoid forcing food or drink if swallowing is unsafe or vomiting is persistent.",
            "caution": "Food cannot cure these infections. Timely assessment and prescribed treatment may be essential; dengue-like illness with bleeding, severe abdominal pain or drowsiness needs urgent care."
        }
    if "heart attack" in name or "paralysis" in name:
        return {
            "title": "Possible emergency: do not rely on diet advice",
            "eat": "Prioritize emergency medical assessment and follow the treating team's instructions.",
            "limit": "Do not delay emergency care to prepare food or try a home remedy.",
            "caution": "If sudden chest pain, sudden weakness, facial droop, speech difficulty or unconsciousness occurs, call emergency services. Do not give food/drink if swallowing may be impaired."
        }
    return {
        "title": "General balanced-food guidance",
        "eat": "If you have no clinician-directed restrictions, choose a varied balanced diet: vegetables and fruit, pulses/beans, whole grains, and suitable protein sources. Drink fluids according to thirst and medical advice.",
        "limit": "Limit frequent sugary drinks, excess salt, and heavily processed or deep-fried foods. Avoid only foods that you know worsen your symptoms; blanket restrictions are not justified by this model match.",
        "caution": "This condition match is not a confirmed diagnosis. Diet advice can differ with pregnancy, age, kidney/heart/liver disease, allergies and medicines; ask a clinician for a personalized plan."
    }


def safe_precautions(disease):
    items = PRECAUTIONS.get(norm(disease), [])
    cleaned = []
    for item in items:
        v = str(item).strip()
        if not v:
            continue
        # Avoid presenting the dataset's potentially risky medication advice as instruction.
        if "stop taking drug" in v.lower() or "stop medication" in v.lower():
            v = "If you suspect a medication reaction, contact a clinician promptly; do not change prescribed medicines without medical advice."
        cleaned.append(v)
    return cleaned

@app.get("/")
def index():
    return render_template("index.html", symptoms=SYMPTOMS)

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "diseases": len(model.classes_), "symptom_features": len(SYMPTOMS)})

@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    selected = {norm(x) for x in payload.get("symptoms", []) if norm(x) in set(SYMPTOMS)}
    free_text = str(payload.get("free_text", "")).strip()
    selected |= map_free_text_to_symptoms(free_text)
    body_part = str(payload.get("body_part", "General"))
    severity = str(payload.get("severity", "Mild"))

    if not selected:
        return jsonify({
            "error": "Select at least one symptom from the list, or describe symptoms using terms available in this dataset."
        }), 400

    urgent, safety_message = safety_assessment(list(selected), free_text, severity)
    vector = mlb.transform([sorted(selected)])
    scores = model.predict_proba(vector)[0]
    top_idx = np.argsort(scores)[::-1][:3]
    predictions = []
    for i in top_idx:
        disease = str(model.classes_[i]).strip()
        predictions.append({
            "condition": disease,
            "model_score": round(float(scores[i]) * 100, 1),
            "description": DESCRIPTIONS.get(norm(disease), "No description is available in the supplied description file."),
            "dataset_precautions": safe_precautions(disease),
            "ayurveda_guidance": ayurveda_guidance(disease, list(selected), severity, urgent),
            "diet_guidance": diet_guidance(disease, urgent, severity)
        })
    specialist = specialist_for(list(selected), body_part)
    return jsonify({
        "matched_symptoms": sorted(selected),
        "predictions": predictions,
        "specialist": specialist,
        "activity": activity_guidance(list(selected), body_part, severity, free_text),
        "urgent": urgent,
        "safety_message": safety_message,
        "disclaimer": (
            "Model scores are not clinical probabilities. The supplied dataset may contain "
            "synthetic/permuted examples and has not been clinically validated. This is not a diagnosis. "
            "Dataset precautions are reference material, not personalized medical advice."
        )
    })

@app.post("/api/doctors")
def doctors():
    payload = request.get_json(silent=True) or {}
    location = str(payload.get("location", "")).strip()
    specialist = str(payload.get("specialist", "General Physician")).strip()
    if not location:
        return jsonify({"error": "Enter a city or PIN code."}), 400

    maps_url = "https://www.google.com/maps/search/?api=1&query=" + requests.utils.quote(
        specialist + " near " + location
    )
    if not GOOGLE_PLACES_API_KEY:
        return jsonify({"results": [], "fallback": maps_url,
                        "message": "Google Places API key is not configured. Use Google Maps search."})

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,places.nationalPhoneNumber,"
            "places.websiteUri,places.rating,places.googleMapsUri"
        )
    }
    body = {"textQuery": specialist + " near " + location, "pageSize": 8, "languageCode": "en"}
    try:
        response = requests.post(url, headers=headers, json=body, timeout=20)
        if response.status_code >= 400:
            return jsonify({"error": "Places API request failed. Check API enablement, key restrictions, and billing.",
                            "details": response.text[:400], "fallback": maps_url}), 502
        data = response.json()
        results = []
        for p in data.get("places", []):
            results.append({
                "name": p.get("displayName", {}).get("text", "Clinic"),
                "address": p.get("formattedAddress", "Address not listed"),
                "phone": p.get("nationalPhoneNumber", ""),
                "website": p.get("websiteUri", ""),
                "rating": p.get("rating"),
                "maps": p.get("googleMapsUri", "")
            })
        return jsonify({"results": results, "fallback": maps_url})
    except requests.RequestException:
        return jsonify({"error": "Could not contact Google Places API.", "fallback": maps_url}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
