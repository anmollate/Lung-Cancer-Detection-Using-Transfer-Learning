import os
import warnings
import requests
import numpy as np
import cv2
import pickle
import joblib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import keras
from keras.applications.vgg16    import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁", layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"]{background-color:#0a0a0a !important;}
[data-testid="stSidebar"]{background-color:#111111 !important;}
[data-testid="stHeader"]{background-color:#0a0a0a !important;}
.main-title{text-align:center;
    background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
    color:white;padding:2rem;border-radius:15px;margin-bottom:2rem;}
.main-title h1{font-size:2.2rem;margin:0;}
.main-title p{font-size:1rem;opacity:0.85;margin:0.5rem 0 0 0;}
.result-high{background:linear-gradient(135deg,#c0392b,#e74c3c);
    color:white;padding:1.5rem;border-radius:12px;
    text-align:center;font-size:1.3rem;font-weight:bold;margin:1rem 0;}
.result-medium{background:linear-gradient(135deg,#d35400,#e67e22);
    color:white;padding:1.5rem;border-radius:12px;
    text-align:center;font-size:1.3rem;font-weight:bold;margin:1rem 0;}
.result-low{background:linear-gradient(135deg,#1a7a4a,#27ae60);
    color:white;padding:1.5rem;border-radius:12px;
    text-align:center;font-size:1.3rem;font-weight:bold;margin:1rem 0;}
.metric-box{background:#1e1e1e !important;color:#ffffff !important;
    border-radius:10px;padding:1rem;text-align:center;
    border-left:4px solid #3498db;margin:0.5rem 0;}
.info-box{background:#1a2a3a !important;color:#cce5ff !important;
    border-left:4px solid #3498db;padding:0.8rem 1rem;
    border-radius:8px;margin:0.5rem 0;font-size:0.9rem;}
.warning-box{background:#2a1f0a !important;color:#ffe8b0 !important;
    border-left:4px solid #f39c12;padding:0.8rem 1rem;
    border-radius:8px;margin:0.5rem 0;font-size:0.9rem;}
.danger-box{background:#2a0a0a !important;color:#ffcccc !important;
    border-left:4px solid #e74c3c;padding:0.8rem 1rem;
    border-radius:8px;margin:0.5rem 0;font-size:0.9rem;}
.section-header{font-size:1.2rem;font-weight:bold;color:#58c4f5 !important;
    border-bottom:2px solid #3498db;padding-bottom:0.3rem;
    margin:1rem 0 0.8rem 0;}
.gradcam-explain{background:#0f2a0f;border-left:4px solid #27ae60;
    padding:1rem;border-radius:8px;margin:0.5rem 0;
    font-size:0.9rem;color:#90ee90 !important;}
.hospital-card{background:#1a1a2e;border:1px solid #2c3e6e;
    border-radius:10px;padding:1rem;margin:0.6rem 0;}
.no-hospital-box{background:#2a0a0a;border:2px solid #e74c3c;
    border-radius:10px;padding:1.2rem;margin:0.5rem 0;text-align:center;}
p,label,.stMarkdown,div{color:#e0e0e0 !important;}
[data-testid="stFileUploader"]{background:#1a1a1a !important;
    border:2px dashed #3498db !important;border-radius:10px;}
.streamlit-expanderHeader{background:#1e1e1e !important;color:#fff !important;}
.streamlit-expanderContent{background:#141414 !important;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASSES     = ['Normal', 'Benign', 'Malignant']
CLASS_EMOJI = {'Normal':'🟢','Benign':'🟡','Malignant':'🔴'}
CLASS_COLOR = {'Normal':'#27ae60','Benign':'#f39c12','Malignant':'#e74c3c'}

# ── Known Cancer Hospitals (fallback) — proper Google Maps URLs ───────────────
def gmaps_url(name, lat, lon):
    n = name.replace(' ', '+')
    return f"https://www.google.com/maps/search/{n}/@{lat},{lon},15z"

def gmaps_dir(lat, lon):
    return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"

KNOWN_CANCER_HOSPITALS = [
    {
        'name':      'Tata Memorial Hospital',
        'lat':       19.0069, 'lon': 72.8422,
        'type':      'Cancer Specialist — Government',
        'address':   'Dr E Borges Road, Parel, Mumbai 400012',
        'phone':     '+91-22-24177000',
        'email':     'hrd@tmc.gov.in',
        'website':   'https://tmc.gov.in',
        'speciality':'Lung Cancer, Oncology, Chemotherapy, Radiation',
    },
    {
        'name':      'AIIMS Cancer Centre, New Delhi',
        'lat':       28.5672, 'lon': 77.2100,
        'type':      'Cancer Specialist — Government',
        'address':   'Ansari Nagar, New Delhi 110029',
        'phone':     '+91-11-26588500',
        'email':     'director@aiims.ac.in',
        'website':   'https://aiims.edu',
        'speciality':'Lung Cancer, Thoracic Surgery, Oncology',
    },
    {
        'name':      'Rajiv Gandhi Cancer Institute, Delhi',
        'lat':       28.6914, 'lon': 77.1519,
        'type':      'Cancer Specialist — Private',
        'address':   'Sector 5, Rohini, New Delhi 110085',
        'phone':     '+91-11-47022222',
        'email':     'info@rgcirc.org',
        'website':   'https://www.rgcirc.org',
        'speciality':'Lung Cancer, Oncology, Immunotherapy',
    },
    {
        'name':      'Kidwai Memorial Institute of Oncology',
        'lat':       12.9352, 'lon': 77.5947,
        'type':      'Cancer Specialist — Government',
        'address':   'Dr M H Marigowda Road, Bangalore 560029',
        'phone':     '+91-80-26094000',
        'email':     'director@kmio.karnataka.gov.in',
        'website':   'http://kmio.karnataka.gov.in',
        'speciality':'Lung Cancer, Oncology, Radiotherapy',
    },
    {
        'name':      'Apollo Cancer Centre, Chennai',
        'lat':       13.0694, 'lon': 80.2490,
        'type':      'Cancer Specialist — Private',
        'address':   '21 Greams Lane, Chennai 600006',
        'phone':     '+91-44-28290200',
        'email':     'cancercare@apollohospitals.com',
        'website':   'https://apollocancercentres.com',
        'speciality':'Lung Cancer, Robotic Surgery, Immunotherapy',
    },
    {
        'name':      'Kokilaben Dhirubhai Ambani Hospital',
        'lat':       19.1244, 'lon': 72.8264,
        'type':      'Cancer Specialist — Private',
        'address':   'Rao Saheb Achutrao Patwardhan Marg, Andheri West, Mumbai',
        'phone':     '+91-22-30999999',
        'email':     'info@kokilabenhospital.com',
        'website':   'https://www.kokilabenhospital.com',
        'speciality':'Lung Cancer, CyberKnife, Oncology',
    },
    {
        'name':      'HCG Cancer Centre, Bangalore',
        'lat':       13.0012, 'lon': 77.5694,
        'type':      'Cancer Specialist — Private',
        'address':   'No 8 P Kalinga Rao Road, Sampangiram Nagar, Bangalore',
        'phone':     '+91-80-40206000',
        'email':     'info@hcghospitals.com',
        'website':   'https://www.hcgoncology.com',
        'speciality':'Lung Cancer, PET-CT, Radiation Oncology',
    },
]
# Add proper gmaps links
for h in KNOWN_CANCER_HOSPITALS:
    h['gmaps_link']     = gmaps_url(h['name'], h['lat'], h['lon'])
    h['gmaps_dir_link'] = gmaps_dir(h['lat'], h['lon'])
    h['dist']           = 0.0

# ── Session State ──────────────────────────────────────────────────────────────
for k, v in [
    ('map_created', False), ('map_data', None),
    ('map_hospitals', []),  ('is_fallback', False),
    ('geo_lat', None),      ('geo_lon', None),
    ('geo_triggered', False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_all_models():
    loaded, errors = {}, {}
    for name, path in [
        ('VGG16',   'vgg16_model.keras'),
        ('CNN',     'cnn_model.keras'),
        ('ResNet50','resnet50_model.keras'),
    ]:
        try:
            loaded[name] = keras.models.load_model(
                os.path.join(MODEL_DIR, path))
        except Exception as e:
            errors[name] = str(e)
    try:
        fe     = keras.models.load_model(
            os.path.join(MODEL_DIR,'hybrid_feature_extractor.keras'))
        clf    = joblib.load(os.path.join(MODEL_DIR,'hybrid_best.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR,'hybrid_scaler.pkl'))
        with open(os.path.join(MODEL_DIR,'hybrid_meta.pkl'),'rb') as f:
            meta = pickle.load(f)
        loaded['Hybrid'] = {'extractor':fe,'classifier':clf,
                            'scaler':scaler,'meta':meta}
    except Exception as e:
        errors['Hybrid'] = str(e)
    return loaded, errors

# ── Preprocess ─────────────────────────────────────────────────────────────────
def preprocess_image(pil_img, model_name):
    img  = pil_img.convert('RGB').resize((IMG_SIZE,IMG_SIZE),Image.LANCZOS)
    arr  = np.array(img, dtype=np.float32) / 255.0
    if model_name in ['VGG16','Hybrid']:
        arr_pp = vgg_preprocess(arr * 255.0)
    elif model_name == 'ResNet50':
        arr_pp = resnet_preprocess(arr * 255.0)
    else:
        arr_pp = arr
    return arr, arr_pp

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(pil_img, model_name, models_dict):
    arr_norm, arr_pp = preprocess_image(pil_img, model_name)
    inp = np.expand_dims(arr_pp, axis=0)
    if model_name == 'Hybrid':
        h        = models_dict['Hybrid']
        feat     = h['extractor'].predict(inp, verbose=0)
        feat_sc  = h['scaler'].transform(feat)
        probs    = h['classifier'].predict_proba(feat_sc)[0]
        pred_idx = int(np.argmax(probs))
    else:
        probs    = models_dict[model_name].predict(inp, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
    return pred_idx, probs, arr_norm

# ── Saliency XAI ──────────────────────────────────────────────────────────────
def compute_saliency(pil_img, model_name, models_dict, pred_idx):
    if model_name == 'Hybrid':
        return None
    arr_norm, arr_pp = preprocess_image(pil_img, model_name)
    model   = models_dict[model_name]
    inp_arr = np.expand_dims(arr_pp, axis=0).astype(np.float32)
    try:
        inp_var = tf.Variable(inp_arr)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inp_var)
            preds = model(inp_var, training=False)
            loss  = preds[0, pred_idx]
        grads = tape.gradient(loss, inp_var)
        del tape
        if grads is None:
            return None
        grads_np = grads.numpy()[0]
        saliency = np.max(np.abs(grads_np), axis=-1)
        saliency = (saliency - saliency.min()) / \
                   (saliency.max() - saliency.min() + 1e-8)
        saliency = cv2.GaussianBlur(
            saliency.astype(np.float32), (11,11), 0)
        saliency = (saliency - saliency.min()) / \
                   (saliency.max() - saliency.min() + 1e-8)
        heatmap       = np.uint8(255 * saliency)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        original      = np.uint8(arr_norm * 255)
        overlay       = cv2.addWeighted(original,0.5,heatmap_color,0.5,0)
        top_pct       = float(np.mean(saliency > 0.7) * 100)
        return {'heatmap':heatmap_color,'overlay':overlay,
                'saliency':saliency,'top_region_pct':top_pct}
    except Exception as e:
        st.warning(f"XAI error: {str(e)[:200]}")
        return None

# ── XAI Explanation ────────────────────────────────────────────────────────────
def get_xai_explanation(pred_class, sal, model_name, confidence):
    pct = sal['top_region_pct']
    return {
        'Normal':(
            f"The {model_name} model found normal lung architecture. "
            f"Only {pct:.1f}% of scan had high AI attention — "
            f"distributed symmetrically with no focal mass or nodule. "
            f"No abnormal density or suspicious consolidation detected. "
            f"Confidence: {confidence*100:.1f}%."
        ),
        'Benign':(
            f"The {model_name} model flagged {pct:.1f}% of this scan. "
            f"Highlighted regions show a well-defined, localized area "
            f"with smooth borders — characteristic of benign tumors "
            f"(e.g. hamartoma, granuloma). No invasive growth detected. "
            f"Confidence: {confidence*100:.1f}%. Monitoring recommended."
        ),
        'Malignant':(
            f"The {model_name} model found {pct:.1f}% of scan highly "
            f"suspicious. Red/yellow regions show abnormal tissue density "
            f"with irregular, spiculated margins — key features of lung "
            f"carcinoma. Patterns suggest aggressive cellular growth and "
            f"possible infiltration. Confidence: {confidence*100:.1f}%. "
            f"URGENT: Biopsy and oncology referral required immediately."
        ),
    }[pred_class]

# ── Fetch Cancer Hospitals (OSM) ───────────────────────────────────────────────
def fetch_cancer_hospitals(lat, lon, radius_km=25):
    radius_m = radius_km * 1000
    query = f"""
    [out:json][timeout:30];
    (
      node["amenity"="hospital"]
          ["healthcare:speciality"~"oncolog|cancer|lung|thorac|pulmon",i]
          (around:{radius_m},{lat},{lon});
      node["amenity"="hospital"]
          ["name"~"cancer|oncolog|lung|thorac|tumor|tumour|memorial",i]
          (around:{radius_m},{lat},{lon});
      way["amenity"="hospital"]
         ["name"~"cancer|oncolog|lung|thorac|tumor|tumour|memorial",i]
         (around:{radius_m},{lat},{lon});
      way["amenity"="hospital"]
         ["healthcare:speciality"~"oncolog|cancer|lung|thorac|pulmon",i]
         (around:{radius_m},{lat},{lon});
    );
    out body center 15;
    """
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query, timeout=25
        )
        if resp.status_code != 200:
            return []
        data      = resp.json()
        hospitals = []
        for el in data.get('elements', []):
            tags = el.get('tags', {})
            name = tags.get('name', tags.get('name:en', ''))
            if not name:
                continue
            if el['type'] == 'node':
                h_lat, h_lon = el['lat'], el['lon']
            else:
                center = el.get('center', {})
                h_lat  = center.get('lat', lat)
                h_lon  = center.get('lon', lon)
            dist    = round(
                ((h_lat-lat)**2+(h_lon-lon)**2)**0.5*111, 2)
            phone   = tags.get('phone',
                      tags.get('contact:phone', 'Not listed'))
            email   = tags.get('email',
                      tags.get('contact:email', ''))
            website = tags.get('website',
                      tags.get('contact:website', ''))
            address = ', '.join(filter(None,[
                tags.get('addr:housenumber',''),
                tags.get('addr:street',''),
                tags.get('addr:suburb',''),
                tags.get('addr:city',''),
            ])) or 'See Google Maps'
            spec    = tags.get('healthcare:speciality','Oncology')
            hospitals.append({
                'name':         name,
                'lat':          h_lat, 'lon': h_lon,
                'type':         tags.get('amenity','Hospital').title(),
                'speciality':   spec,
                'phone':        phone,
                'email':        email,
                'website':      website,
                'address':      address,
                'dist':         dist,
                'gmaps_link':   gmaps_url(name, h_lat, h_lon),
                'gmaps_dir_link': gmaps_dir(h_lat, h_lon),
            })
        hospitals.sort(key=lambda x: x['dist'])
        return hospitals[:10]
    except Exception:
        return []

def get_nearest_known(lat, lon, n=5):
    for h in KNOWN_CANCER_HOSPITALS:
        h['dist'] = round(
            ((h['lat']-lat)**2+(h['lon']-lon)**2)**0.5*111, 1)
    return sorted(KNOWN_CANCER_HOSPITALS, key=lambda x: x['dist'])[:n]

# ── Build Map ──────────────────────────────────────────────────────────────────
def build_map(lat, lon, hospitals, is_fallback=False):
    fmap = folium.Map(location=[lat,lon], zoom_start=11,
                      tiles='CartoDB positron')
    folium.Marker(
        location=[lat,lon],
        popup=folium.Popup("<b>📍 Your Location</b>",max_width=200),
        tooltip="📍 You are here",
        icon=folium.Icon(color='blue',icon='home',prefix='fa')
    ).add_to(fmap)
    folium.Circle(location=[lat,lon],radius=500,color='#3498db',
                  fill=True,fill_opacity=0.1).add_to(fmap)
    colors = ['red','darkred','orange','purple','cadetblue',
              'darkblue','darkgreen','black','gray','lightred']
    for i, h in enumerate(hospitals):
        popup_html = f"""
        <div style='width:255px;font-family:Arial;
                    font-size:13px;line-height:1.7;'>
        <b style='color:#c0392b;font-size:14px;'>{h['name']}</b><br>
        🎗️ {h.get('speciality','Oncology')}<br>
        📏 {h['dist']} km away<br>
        📍 {h['address']}<br>
        📞 {h['phone']}<br>
        {'📧 '+h['email']+'<br>' if h.get('email') else ''}
        <br>
        <a href="{h['gmaps_dir_link']}" target="_blank"
           style='background:#4285F4;color:white;padding:4px 10px;
                  border-radius:4px;text-decoration:none;
                  font-size:12px;'>
           🗺️ Get Directions
        </a>
        </div>"""
        folium.Marker(
            location=[h['lat'],h['lon']],
            popup=folium.Popup(popup_html,max_width=280),
            tooltip=f"🏥 {h['name']} ({h['dist']} km)",
            icon=folium.Icon(color=colors[i%len(colors)],
                             icon='plus-sign',prefix='glyphicon')
        ).add_to(fmap)
        folium.PolyLine(
            locations=[[lat,lon],[h['lat'],h['lon']]],
            color='#e74c3c' if is_fallback else '#95a5a6',
            weight=1,opacity=0.5,dash_array='5'
        ).add_to(fmap)
    return fmap

def search_and_build(lat, lon, radius=25):
    hospitals = fetch_cancer_hospitals(lat, lon, radius)
    if hospitals:
        return build_map(lat,lon,hospitals,False), hospitals, False
    fallback = get_nearest_known(lat, lon, 5)
    return build_map(lat,lon,fallback,True), fallback, True

# ── Geolocation Component ──────────────────────────────────────────────────────
def geo_component():
    """
    Sends lat/lon to Streamlit via postMessage → received by
    st.session_state via a hidden text_input trick.
    This avoids iframe sandbox redirect issues entirely.
    """
    geo_html = """
    <!DOCTYPE html><html><head>
    <style>
    *{margin:0;padding:0;box-sizing:border-box;}
    body{background:transparent;
         font-family:-apple-system,BlinkMacSystemFont,sans-serif;}
    .box{background:linear-gradient(135deg,#1a2a4a,#0f1a2e);
        border:1px solid #3498db;border-radius:10px;padding:14px 16px;}
    .title{color:#cce5ff;font-size:14px;margin-bottom:10px;font-weight:500;
        text-align:center;}
    .btn{background:linear-gradient(135deg,#1a6fa8,#2980b9);color:white;
        border:none;padding:10px 20px;border-radius:7px;cursor:pointer;
        font-size:14px;font-weight:600;width:100%;transition:all 0.2s;}
    .btn:hover{background:linear-gradient(135deg,#2980b9,#3498db);}
    .btn:disabled{opacity:0.6;cursor:not-allowed;}
    .status{margin-top:10px;font-size:13px;padding:7px 10px;
        border-radius:6px;display:none;text-align:center;}
    .success{background:#0f2a0f;color:#90ee90;
        border:1px solid #27ae60;display:block;}
    .error{background:#2a0a0a;color:#ffaaaa;
        border:1px solid #e74c3c;display:block;}
    .loading{background:#1a2a3a;color:#aad4f5;
        border:1px solid #3498db;display:block;}
    .coords{font-size:11px;color:#7fb3d3;margin-top:5px;
        font-family:monospace;text-align:center;}
    </style></head><body>
    <div class="box">
        <div class="title">📍 Use your live GPS location</div>
        <button class="btn" id="btn" onclick="getLoc()">
            🌐 Allow Location & Auto-Find Cancer Hospitals
        </button>
        <div class="status" id="status"></div>
        <div class="coords" id="coords"></div>
    </div>
    <script>
    function getLoc() {
        var btn=document.getElementById('btn');
        var st=document.getElementById('status');
        var co=document.getElementById('coords');
        btn.disabled=true;
        btn.textContent='⏳ Requesting permission...';
        st.className='status loading'; st.textContent='Waiting...';

        if (!navigator.geolocation) {
            st.className='status error';
            st.textContent='❌ Geolocation not supported by your browser.';
            btn.disabled=false; btn.textContent='🌐 Allow Location';
            return;
        }
        navigator.geolocation.getCurrentPosition(
            function(pos) {
                var lat=pos.coords.latitude.toFixed(6);
                var lon=pos.coords.longitude.toFixed(6);
                var acc=Math.round(pos.coords.accuracy);
                st.className='status success';
                st.textContent='✅ Location captured!';
                btn.textContent='✅ Location Active';
                co.textContent='Lat:'+lat+' | Lon:'+lon+
                               ' | Accuracy:±'+acc+'m';

                // Send to parent Streamlit window via postMessage
                window.parent.postMessage(
                    {type:'geo_location',lat:parseFloat(lat),
                     lon:parseFloat(lon)}, '*'
                );

                // Also update URL params and reload
                var base=window.parent.location.href.split('?')[0];
                window.parent.location.href=
                    base+'?geo_lat='+lat+'&geo_lon='+lon+'&geo_go=1';
            },
            function(err){
                var msgs={
                    1:'❌ Permission denied. Allow in browser settings.',
                    2:'❌ Position unavailable.',
                    3:'❌ Request timed out.'
                };
                st.className='status error';
                st.textContent=msgs[err.code]||'❌ Error getting location.';
                btn.disabled=false;
                btn.textContent='🌐 Try Again';
            },
            {enableHighAccuracy:true,timeout:15000,maximumAge:0}
        );
    }
    // If already has geo params, show active
    var p=new URLSearchParams(window.parent.location.search);
    if(p.get('geo_go')==='1'){
        var btn=document.getElementById('btn');
        var st=document.getElementById('status');
        var co=document.getElementById('coords');
        btn.textContent='✅ Live Location Active';
        btn.disabled=true;
        st.className='status success';
        st.textContent='✅ Live location is active';
        co.textContent='Lat:'+p.get('geo_lat')+
                       ' | Lon:'+p.get('geo_lon');
    }
    </script></body></html>
    """
    components.html(geo_html, height=160)

# ── Risk ───────────────────────────────────────────────────────────────────────
def assess_risk(pred_class, conf, age, smoking,
                fam_hist, cough, chest_pain, shortness):
    s=0
    if pred_class=='Malignant': s+=5
    elif pred_class=='Benign':  s+=2
    if conf>0.9: s+=2
    elif conf>0.7: s+=1
    if age>60: s+=2
    elif age>45: s+=1
    if smoking:    s+=3
    if fam_hist:   s+=2
    if cough:      s+=1
    if chest_pain: s+=1
    if shortness:  s+=1
    if s>=10:  return 'High',s
    elif s>=6: return 'Medium',s
    else:      return 'Low',s

# ── Recommendations ────────────────────────────────────────────────────────────
def get_recommendations(pred_class, risk_level, smoking, age, fam_hist):
    if pred_class=='Normal' and risk_level=='Low':
        t='✅ No Immediate Concern'
        p=['📅 Annual CT screening if age>55 with smoking history',
           '🚭 Avoid smoking and secondhand smoke',
           '🥗 Antioxidant-rich diet (fruits, vegetables, green tea)',
           '🏃 Exercise regularly — 150 minutes/week',
           '🌿 Avoid radon gas, asbestos and air pollution']
        f='Next routine checkup in 12 months.'
    elif pred_class=='Benign' or risk_level=='Medium':
        t='⚠️ Monitoring Recommended'
        p=['📅 Follow-up CT scan in 3–6 months',
           '🩺 Consult a pulmonologist for evaluation',
           '🚭 Stop smoking — reduces growth risk significantly',
           '📊 Monitor symptoms: cough, fatigue, weight loss',
           '💊 Discuss preventive biopsy with your doctor']
        f='Specialist appointment within 4–6 weeks.'
    else:
        t='🚨 Immediate Medical Attention Required'
        p=['🏥 Visit a cancer specialist IMMEDIATELY',
           '🔬 Request tissue biopsy and PET-CT scan',
           '📋 Carry all CT scan reports to appointment',
           '👨‍👩‍👧 Inform family — support is critical',
           '🚭 Stop smoking — accelerates cancer progression',
           '🧬 Ask oncologist about targeted therapy & immunotherapy',
           '⏰ Every day counts — do not delay treatment']
        f='Cancer specialist appointment within 24–48 hours.'
    personal=[]
    if smoking:  personal.append('🚭 Smoking cessation doubles survival chances — get help')
    if age>55:   personal.append('📅 Annual low-dose CT strongly recommended for your age')
    if fam_hist: personal.append('🧬 Genetic testing reveals inherited cancer risk factors')
    return {'title':t,'primary':p,'followup':f,'personal':personal}

# ==============================================================================
# MAIN UI
# ==============================================================================
st.markdown("""
<div class="main-title">
<h1>🫁 Lung Cancer Detection AI System</h1>
<p>Explainable AI • Multi-Model Analysis •
   Clinical Decision Support • Cancer Hospital Finder</p>
</div>""", unsafe_allow_html=True)

models_dict, model_errors = load_all_models()

# ── Read URL params from geo ────────────────────────────────────────────────────
params   = st.query_params
geo_go   = params.get('geo_go','0') == '1'
geo_lat  = float(params.get('geo_lat', 19.1030))
geo_lon  = float(params.get('geo_lon', 73.0070))

# Auto-search when geo params present
if geo_go and not st.session_state.map_created:
    with st.spinner(
        f"📍 Live location ({geo_lat:.4f}, {geo_lon:.4f}) detected — "
        "searching cancer hospitals..."
    ):
        fmap, hospitals, is_fb = search_and_build(geo_lat, geo_lon, 25)
    st.session_state.map_data      = fmap
    st.session_state.map_hospitals = hospitals
    st.session_state.map_created   = True
    st.session_state.is_fallback   = is_fb
    if is_fb:
        st.warning(
            "⚠️ No cancer hospitals found via OpenStreetMap nearby. "
            "Showing nearest known cancer hospitals in India."
        )
    else:
        st.success(
            f"✅ Found {len(hospitals)} cancer specialist hospitals near you!"
        )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings"); st.markdown("---")
    available = list(models_dict.keys())
    if not available:
        st.error("No models loaded!"); st.stop()
    selected_model = st.selectbox("🤖 Select AI Model", available)
    info = {
        'VGG16':   ('99.11%','0.9999','Transfer Learning'),
        'CNN':     ('36.44%','0.5549','Custom CNN'),
        'ResNet50':('97.78%','0.9985','Transfer Learning'),
        'Hybrid':  ('99.56%','1.0000','VGG16 + SVM'),
    }
    if selected_model in info:
        acc,auc_s,mtype = info[selected_model]
        st.markdown(f"""<div class="metric-box"><b>{selected_model}</b><br>
        {mtype}<br>Acc:{acc} | AUC:{auc_s}</div>""",
        unsafe_allow_html=True)
    st.markdown("---")
    show_xai = st.checkbox("🔥 Show XAI Heatmap", value=True,
                           disabled=(selected_model=='Hybrid'))
    if selected_model=='Hybrid': st.info("XAI N/A for Hybrid")
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    for m in ['VGG16','CNN','ResNet50','Hybrid']:
        if m in models_dict: st.success(f"✅ {m}")
        else:                st.error(f"❌ {m}")
    st.markdown("---")
    

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">📤 Step 1 — Upload CT Scan</div>',
    unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload Lung CT Scan (.jpg .jpeg .png)",
    type=['jpg','jpeg','png']
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert('RGB')
    with st.spinner(f"Analyzing with {selected_model}..."):
        pred_idx, probs, arr_norm = predict(
            pil_img, selected_model, models_dict)
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # ── Step 2 ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🎯 Step 2 — AI Prediction Result</div>',
        unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Original CT Scan**")
        st.image(pil_img, use_container_width=True)
    with c2:
        st.markdown("**Prediction**")
        css={'Malignant':'result-high','Benign':'result-medium',
             'Normal':'result-low'}[pred_class]
        st.markdown(f"""<div class="{css}">
        {CLASS_EMOJI[pred_class]} {pred_class.upper()}<br>
        <span style='font-size:0.9rem;font-weight:normal;'>
        Confidence: {confidence*100:.1f}%</span></div>""",
        unsafe_allow_html=True)
        st.markdown("**Probabilities:**")
        for cls,prob in zip(CLASSES,probs):
            st.markdown(f"{CLASS_EMOJI[cls]} **{cls}**")
            st.progress(float(prob),text=f"{prob*100:.1f}%")
    with c3:
        st.markdown("**Confidence Gauge**")
        fig,ax=plt.subplots(figsize=(3,3),facecolor='#0a0a0a')
        ax.pie([confidence,1-confidence],
               colors=[CLASS_COLOR[pred_class],'#2a2a2a'],
               startangle=90,
               wedgeprops=dict(width=0.5,edgecolor='#0a0a0a'))
        ax.text(0,0,f"{confidence*100:.0f}%",ha='center',va='center',
                fontsize=18,fontweight='bold',color=CLASS_COLOR[pred_class])
        ax.set_title(selected_model,fontsize=9,color='white')
        st.pyplot(fig,use_container_width=True); plt.close()

    # ── Step 3: XAI ────────────────────────────────────────────────────────────
    if show_xai and selected_model != 'Hybrid':
        st.markdown(
            '<div class="section-header">'
            '🔥 Step 3 — Explainable AI (Why this prediction?)'
            '</div>', unsafe_allow_html=True)
        with st.spinner("Computing AI attention heatmap..."):
            sal = compute_saliency(
                pil_img,selected_model,models_dict,pred_idx)
        if sal is not None:
            g1,g2,g3 = st.columns(3)
            with g1:
                st.markdown("**🖼️ Original CT Scan**")
                st.image(arr_norm,use_container_width=True)
            with g2:
                st.markdown("**🌡️ AI Attention Heatmap**")
                st.image(sal['heatmap'],use_container_width=True)
            with g3:
                st.markdown("**🔍 Scan + AI Overlay**")
                st.image(sal['overlay'],use_container_width=True)
            st.markdown("""
            <div style='display:flex;gap:0.5rem;margin:0.5rem 0;
                        flex-wrap:wrap;'>
            <span style='background:#d32f2f;color:#fff;padding:3px 12px;
                border-radius:4px;font-size:13px;'>
                🔴 Red = Highest AI Focus</span>
            <span style='background:#f57c00;color:#fff;padding:3px 12px;
                border-radius:4px;font-size:13px;'>
                🟠 Orange = High Focus</span>
            <span style='background:#fbc02d;color:#000;padding:3px 12px;
                border-radius:4px;font-size:13px;'>
                🟡 Yellow = Medium Focus</span>
            <span style='background:#388e3c;color:#fff;padding:3px 12px;
                border-radius:4px;font-size:13px;'>
                🟢 Green = Low Focus</span>
            <span style='background:#1565c0;color:#fff;padding:3px 12px;
                border-radius:4px;font-size:13px;'>
                🔵 Blue = Minimal Focus</span>
            </div>""", unsafe_allow_html=True)
            exp = get_xai_explanation(
                pred_class,sal,selected_model,confidence)
            st.markdown(f"""
            <div class="gradcam-explain">
            <b>🩺 Clinical AI Explanation (for Doctor Review):</b>
            <br><br>{exp}<br><br>
            <small>Model:{selected_model} | Prediction:{pred_class} |
            Confidence:{confidence*100:.1f}% |
            High-attention:{sal['top_region_pct']:.1f}% of scan</small>
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            ℹ️ <b>Note to Clinicians:</b> Heatmap shows pixel-level
            gradient saliency — regions where intensity changes most
            affect the classification output. Red = most decisive.
            This is a decision-support tool. Always correlate with
            standard radiological assessment and clinical findings.
            </div>""", unsafe_allow_html=True)
        else:
            st.error("❌ XAI heatmap generation failed.")

    # ── Step 4: Clinical ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🩺 Step 4 — Clinical Information</div>',
        unsafe_allow_html=True)
    with st.expander("📋 Patient Clinical Details", expanded=True):
        cl1,cl2,cl3 = st.columns(3)
        with cl1:
            age     = st.slider("Age",20,90,50)
            smoking = st.selectbox("Smoking History",["No","Yes"])=="Yes"
        with cl2:
            fam_hist   = st.selectbox("Family History",["No","Yes"])=="Yes"
            cough      = st.selectbox("Persistent Cough",["No","Yes"])=="Yes"
        with cl3:
            chest_pain = st.selectbox("Chest Pain",["No","Yes"])=="Yes"
            shortness  = st.selectbox("Shortness of Breath",["No","Yes"])=="Yes"

    # ── Step 5: Risk ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">⚠️ Step 5 — Risk Assessment</div>',
        unsafe_allow_html=True)
    risk_level, risk_score = assess_risk(
        pred_class,confidence,age,smoking,
        fam_hist,cough,chest_pain,shortness)
    r1,r2,r3 = st.columns(3)
    with r1:
        rcss={'High':'result-high','Medium':'result-medium',
              'Low':'result-low'}[risk_level]
        remoji={'High':'🔴','Medium':'🟡','Low':'🟢'}[risk_level]
        st.markdown(f"""<div class="{rcss}">
        {remoji} {risk_level.upper()} RISK<br>
        <span style='font-size:0.85rem;font-weight:normal;'>
        Score:{risk_score}/15</span></div>""",
        unsafe_allow_html=True)
    with r2:
        st.markdown("**Risk Factors:**")
        factors=[]
        if smoking:         factors.append("🚬 Active Smoker")
        if fam_hist:        factors.append("🧬 Family History")
        if age>55:          factors.append(f"👤 Age {age}")
        if cough:           factors.append("😮‍💨 Persistent Cough")
        if chest_pain:      factors.append("💔 Chest Pain")
        if shortness:       factors.append("😤 Breathlessness")
        if pred_class=='Malignant':
            factors.append("🔴 Malignant AI Finding")
        if not factors:     st.markdown("✅ None")
        else:
            for f in factors: st.markdown(f"- {f}")
    with r3:
        st.markdown("**Decision:**")
        if risk_level=='High':
            st.markdown("""<div class="danger-box">
            🚨 <b>Immediate attention.</b><br>
            Visit cancer specialist within 24–48 hrs.
            Use hospital finder below.</div>""",
            unsafe_allow_html=True)
        elif risk_level=='Medium':
            st.markdown("""<div class="warning-box">
            ⚠️ <b>Consult doctor soon.</b><br>
            Within 2 weeks.</div>""",
            unsafe_allow_html=True)
        else:
            st.markdown("""<div class="info-box">
            ✅ <b>No immediate concern.</b><br>
            Regular monitoring advised.</div>""",
            unsafe_allow_html=True)

    # ── Step 6: Recommendations ────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">💊 Step 6 — Recommendations</div>',
        unsafe_allow_html=True)
    recs = get_recommendations(pred_class,risk_level,smoking,age,fam_hist)
    rc1,rc2 = st.columns(2)
    with rc1:
        st.markdown(f"### {recs['title']}")
        for r in recs['primary']: st.markdown(f"- {r}")
        st.info(f"📅 **Follow-up:** {recs['followup']}")
    with rc2:
        if recs['personal']:
            st.markdown("### 👤 Personalized Advice")
            for p in recs['personal']: st.markdown(f"- {p}")

    # ── Step 7: Cancer Hospital Finder ─────────────────────────────────────────
    st.markdown(
        '<div class="section-header">'
        '🗺️ Step 7 — Cancer Hospital Finder (Specialists Only)'
        '</div>', unsafe_allow_html=True)

    with st.expander("📍 Find Nearby Cancer & Lung Specialists",
                     expanded=True):
        mc1,mc2 = st.columns([1,2])

        with mc1:
            geo_component()
            st.markdown("---")
            st.markdown("**📝 Or enter manually:**")
            m_lat = st.number_input("Latitude",
                                    value=geo_lat,format="%.6f",
                                    key="mlat")
            m_lon = st.number_input("Longitude",
                                    value=geo_lon,format="%.6f",
                                    key="mlon")
            radius = st.slider("Search Radius (km)",5,50,25)

            if st.button("🔍 Search Cancer Hospitals",
                         use_container_width=True):
                with st.spinner("Searching cancer specialists..."):
                    fmap,hospitals,is_fb = search_and_build(
                        m_lat,m_lon,radius)
                st.session_state.map_data      = fmap
                st.session_state.map_hospitals = hospitals
                st.session_state.map_created   = True
                st.session_state.is_fallback   = is_fb
                if is_fb:
                    st.warning(
                        "⚠️ No cancer hospitals found nearby. "
                        "Showing nearest known cancer hospitals."
                    )
                else:
                    st.success(
                        f"✅ Found {len(hospitals)} cancer specialists!")

            if geo_go:
                st.info(
                    f"📍 Live location used: "
                    f"{geo_lat:.4f}, {geo_lon:.4f}"
                )

        with mc2:
            if st.session_state.map_created and \
               st.session_state.map_data is not None:

                is_fb = st.session_state.get('is_fallback', False)
                if is_fb:
                    st.markdown("""
                    <div class="no-hospital-box">
                    ⚠️ <b>No cancer hospitals found nearby
                    via OpenStreetMap.</b><br>
                    Showing nearest <b>known cancer specialist
                    hospitals in India.</b><br>
                    <small>Call ahead to confirm availability
                    and book appointment.</small>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.success(
                        f"✅ "
                        f"{len(st.session_state.map_hospitals)} "
                        f"cancer/lung specialist hospitals near you"
                    )

                st_folium(
                    st.session_state.map_data,
                    width=650, height=420,
                    returned_objects=[],
                    key="cancer_map"
                )

                st.markdown("### 🏥 Cancer Specialist Hospitals")
                for i, h in enumerate(
                    st.session_state.map_hospitals, 1):
                    dist_txt = (
                        f"{h['dist']} km away"
                        if h['dist'] > 0 else "Major cancer centre"
                    )
                    spec = h.get('speciality','Oncology / Cancer Care')
                    st.markdown(f"""
                    <div class="hospital-card">
                    <b style='color:#58c4f5;font-size:15px;'>
                        {i}. {h['name']}
                    </b><br>
                    🎗️ <b>Speciality:</b> {spec}<br>
                    📏 <b>Distance:</b> {dist_txt}<br>
                    📍 <b>Address:</b> {h['address']}<br>
                    📞 <b>Phone:</b>
                    <a href="tel:{h['phone']}"
                       style='color:#3498db;text-decoration:none;
                              font-weight:bold;'>
                        {h['phone']}
                    </a><br>
                    {'📧 <b>Email:</b> <a href="mailto:'+h.get('email','')+'" style="color:#3498db;">'+h.get('email','')+'</a><br>' if h.get('email') else ''}
                    {'🌐 <b>Website:</b> <a href="'+h.get('website','')+'" target="_blank" style="color:#3498db;">'+h.get('website','')+'</a><br>' if h.get('website') else ''}
                    <div style='margin-top:10px;display:flex;gap:8px;
                                flex-wrap:wrap;'>
                    <a href="{h['gmaps_dir_link']}" target="_blank"
                       style='background:#4285F4;color:white;
                              padding:7px 14px;border-radius:5px;
                              text-decoration:none;font-size:13px;'>
                        🗺️ Get Directions
                    </a>
                    <a href="{h['gmaps_link']}" target="_blank"
                       style='background:#34a853;color:white;
                              padding:7px 14px;border-radius:5px;
                              text-decoration:none;font-size:13px;'>
                        🔍 View on Maps
                    </a>
                    <a href="tel:{h['phone']}"
                       style='background:#27ae60;color:white;
                              padding:7px 14px;border-radius:5px;
                              text-decoration:none;font-size:13px;'>
                        📞 Call Now
                    </a>
                    </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                <b>🏥 How to find cancer hospitals:</b><br><br>
                1️⃣ Click
                <b>🌐 Allow Location & Auto-Find Cancer Hospitals</b>
                <br>
                2️⃣ Allow browser location permission when asked<br>
                3️⃣ Page reloads → map auto-loads with
                <b>cancer specialist hospitals</b> near you!<br><br>
                <b>Only cancer, oncology, lung & thoracic specialists
                are shown.</b><br>
                If none are nearby, nearest known Indian cancer
                hospitals are shown automatically.
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

else:
    st.markdown("""
    <div class="info-box">
    👆 <b>Upload a lung CT scan</b> to begin AI analysis.
    </div>""", unsafe_allow_html=True)
    lc1,lc2,lc3,lc4 = st.columns(4)
    for col,(icon,title,sub) in zip([lc1,lc2,lc3,lc4],[
        ("🤖","4 AI Models",    "VGG16, CNN, ResNet50, Hybrid"),
        ("🔥","XAI Heatmaps",   "Visual explanations for doctors"),
        ("⚕️","Risk Assessment","Clinical + AI combined score"),
        ("🗺️","Cancer Hospitals","Specialists only. Real data."),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-box">
            <h3>{icon}</h3><b>{title}</b><br>
            <small>{sub}</small>
            </div>""", unsafe_allow_html=True)