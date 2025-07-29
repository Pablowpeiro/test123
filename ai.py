# --- ai.py ---
# Application Streamlit pour aider à planifier des projections de films
# -*- coding: utf-8 -*-

import streamlit as st
import json
import openai
from openai import OpenAI
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.distance import geodesic # Utilise geodesic pour des distances plus précises
import folium
from streamlit_folium import st_folium # Pour mieux intégrer Folium dans Streamlit
import os
import pandas as pd
import uuid
import io # Ajouté pour le buffer Excel en mémoire
import unicodedata
import re

# --- CONFIGURATION DE LA PAGE (DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT) ---
st.set_page_config(layout="wide", page_title="Assistant Cinéma MK2", page_icon="🗺️")

# --- Initialisation des états de session ---
if 'analyse_contexte_done' not in st.session_state:
    st.session_state.analyse_contexte_done = False
if 'recherche_cinemas_done' not in st.session_state:
    st.session_state.recherche_cinemas_done = False
if 'contexte_result' not in st.session_state:
    st.session_state.contexte_result = None
if 'instructions_ia' not in st.session_state:
    st.session_state.instructions_ia = None
if 'reponse_brute_ia' not in st.session_state:
    st.session_state.reponse_brute_ia = None
if 'liste_groupes_resultats' not in st.session_state:
    st.session_state.liste_groupes_resultats = []
if 'dataframes_to_export' not in st.session_state:
    st.session_state.dataframes_to_export = {}
if 'modifications_appliquees' not in st.session_state:
    st.session_state.modifications_appliquees = False

# --- Configuration (Variables globales) ---
GEOCATED_CINEMAS_FILE = "cinemas_groupedBig.json"
GEOCODER_USER_AGENT = "CinemaMapApp/1.0 (App)"
GEOCODER_TIMEOUT = 10

# --- Initialisation du client OpenAI ---
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        st.error("La clé API OpenAI n'a pas été trouvée. Veuillez définir la variable d'environnement OPENAI_API_KEY.")
        st.stop()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client OpenAI : {e}")
    st.stop()

# --- Chargement des données des cinémas pré-géocodées ---
cinemas_ignored_info = None
try:
    with open(GEOCATED_CINEMAS_FILE, "r", encoding="utf-8") as f:
        cinemas_data = json.load(f)
    original_count = len(cinemas_data)
    cinemas_data = [c for c in cinemas_data if c.get('lat') is not None and c.get('lon') is not None]
    valid_count = len(cinemas_data)
    if original_count > valid_count:
        cinemas_ignored_info = f"{original_count - valid_count} cinémas sans coordonnées valides ont été ignorés lors du chargement."
except FileNotFoundError:
    st.error(f"ERREUR : Le fichier de données '{GEOCATED_CINEMAS_FILE}' est introuvable.")
    st.error("Veuillez exécuter le script 'preprocess_cinemas.py' pour générer ce fichier.")
    st.stop()
except json.JSONDecodeError:
    st.error(f"ERREUR : Le fichier de données '{GEOCATED_CINEMAS_FILE}' contient un JSON invalide.")
    st.stop()
except Exception as e:
    st.error(f"Erreur inattendue lors du chargement des données des cinémas : {e}")
    st.stop()

# --- Initialisation du Géocodeur (pour les requêtes utilisateur) ---
geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT, timeout=GEOCODER_TIMEOUT)

# --- Fonctions ---

@st.cache_data(show_spinner=False)
def analyser_requete_ia(question: str):
    """
    Interprète la requête de l'utilisateur en utilisant GPT-4o pour extraire
    les localisations et la fourchette de spectateurs cible.
    Retourne un tuple (liste_instructions, reponse_brute_ia) ou ([], "") en cas d'échec.
    """
    system_prompt = (
        "Tu es un expert en distribution de films en salles en France. L'utilisateur te décrit un projet (test, avant-première, tournée, etc.).\n\n"

        "🎯 Ton objectif : retourner une liste JSON valide de villes avec :\n"
        "- \"localisation\" : une ville en France,\n"
        "- \"nombre\" : nombre de spectateurs à atteindre,\n"
        "- \"nombre_seances\" : (optionnel) nombre de séances prévues.\n\n"

        "🎯 Si l'utilisateur précise un nombre de séances et une fourchette de spectateurs (ex : entre 30 000 et 40 000) :\n"
        "- Choisis un total réaliste dans cette fourchette,\n"
        "- Répartis ce total entre les villes proportionnellement au nombre de séances,\n"
        "- Ne dépasse jamais le maximum, et ne descends jamais en dessous du minimum.\n\n"

        "🎯 Si l'utilisateur précise seulement une fourchette de spectateurs pour une zone :\n"
        "- Choisis un total dans la fourchette,\n"
        "- Répartis les spectateurs équitablement entre les villes de cette zone,\n"
        "- Suppose 1 séance par ville sauf indication contraire.\n\n"

        "🎯 Si plusieurs zones sont mentionnées, génère plusieurs blocs JSON.\n\n"

        "🗺️ Pour les zones vagues, utilise les remplacements suivants :\n"
        "- 'idf', 'île-de-france', 'région parisienne' → ['île-de-france']\n"
        "- 'sud', 'paca', 'sud de la France', 'provence' → ['Marseille', 'Toulouse', 'Nice']\n"
        "- 'nord', 'hauts-de-france' → ['Lille']\n"
        "- 'ouest', 'bretagne', 'normandie' → ['Nantes', 'Rennes', 'Amiens']\n"
        "- 'est', 'grand est', 'alsace' → ['Strasbourg']\n"
        "- 'centre', 'centre-val de loire', 'auvergne' → ['Clermont-Ferrand']\n"
        "- 'France entière', 'toute la France', 'province', 'le territoire', 'le reste du territoire français' → [\n"
        "   'Île-de-france', 'Lille', 'Strasbourg', 'Lyon', 'Marseille', 'Nice',\n"
        "   'Toulouse', 'Montpellier', 'Bordeaux', 'Limoges', 'Nantes', 'Rennes',\n"
        "   'Caen', 'Dijon', 'Clermont-Ferrand', 'Orléans', 'Besançon'\n"
        "]\n\n"

        "💡 Le résultat doit être une **liste JSON strictement valide** :\n"
        "- Format : [{\"localisation\": \"Paris\", \"nombre\": 1000, \"nombre_seances\": 10}]\n"
        "- Utilise des guillemets doubles,\n"
        "- Mets des virgules entre les paires clé/valeur,\n"
        "- Ne retourne **aucun texte en dehors** du JSON.\n\n"

        "💡 Si aucun lieu ni objectif n'est identifiable, retourne simplement : []\n\n"

        "🔐 Règle obligatoire :\n"
        "- Le **nombre total de séances** (addition des \"nombre_seances\") doit correspondre **exactement** à ce que demande l'utilisateur,\n"
        "- Ne t'arrête pas à une distribution ronde ou facile : ajuste si besoin pour que la somme soit strictement exacte."
        "🔐 Règle stricte sur la fourchette :\n"
        "- Si l'utilisateur donne une fourchette de spectateurs (ex : minimum 30 000, maximum 160 000),\n"
        "- Alors le **nombre total de spectateurs** (toutes zones confondues) doit rester **strictement dans cette fourchette**.\n"
        "- Tu ne dois **pas appliquer cette fourchette à une seule zone**, mais à l'ensemble de la demande.\n"
    )

    raw_response = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        raw_response = response.choices[0].message.content.strip()
        try:
            data = json.loads(raw_response)
            if isinstance(data, dict) and "message" in data:
                st.warning(f"⚠️ L'IA a répondu : {data['message']}")
                return [], raw_response
            if isinstance(data, dict) and 'localisation' in data and 'nombre' in data:
                localisation = str(data['localisation']).strip()
                try: nombre = int(data['nombre'])
                except ValueError: nombre = 0
                result = [{"localisation": localisation, "nombre": nombre}]
                if 'nombre_seances' in data:
                    try: result[0]['nombre_seances'] = int(data['nombre_seances'])
                    except (ValueError, TypeError): pass
                return result, raw_response
            elif isinstance(data, list):
                valid_data = []
                all_valid = True
                for item in data:
                    if isinstance(item, dict) and 'localisation' in item and 'nombre' in item:
                        try: item['nombre'] = int(item['nombre'])
                        except (ValueError, TypeError): item['nombre'] = 0; all_valid = False
                        if 'nombre_seances' in item:
                            try: item['nombre_seances'] = int(item['nombre_seances'])
                            except (ValueError, TypeError):
                                if 'nombre_seances' in item: del item['nombre_seances']
                        valid_data.append(item)
                    else:
                        all_valid = False
                if not all_valid:
                     st.warning("Certains éléments retournés par l'IA n'ont pas le format attendu (localisation/nombre).")
                return valid_data, raw_response
            elif isinstance(data, dict):
                potential_keys = ['resultats', 'projections', 'locations', 'intentions', 'data', 'result']
                for key in potential_keys:
                    if key in data and isinstance(data[key], list):
                        extracted = data[key]
                        valid_data = []
                        all_valid = True
                        for item in extracted:
                           if isinstance(item, dict) and 'localisation' in item and 'nombre' in item:
                                try: item['nombre'] = int(item['nombre'])
                                except (ValueError, TypeError): item['nombre'] = 0; all_valid = False
                                if 'nombre_seances' in item:
                                    try: item['nombre_seances'] = int(item['nombre_seances'])
                                    except (ValueError, TypeError):
                                        if 'nombre_seances' in item: del item['nombre_seances']
                                valid_data.append(item)
                           else:
                                all_valid = False
                        if not all_valid:
                             st.warning("Certains éléments (dans un objet) retournés par l'IA n'ont pas le format attendu.")
                        return valid_data, raw_response
                st.warning("L'IA a retourné un objet, mais aucune structure attendue (liste d'intentions) n'a été trouvée.")
                return [], raw_response
            else:
                st.warning("La réponse n'est ni une liste ni un dictionnaire exploitable.")
                return [], raw_response
        except json.JSONDecodeError:
            st.warning("La réponse n'était pas un JSON valide, tentative d'extraction manuelle...")
            try:
                json_part = raw_response[raw_response.find("["):raw_response.rfind("]")+1]
                extracted = json.loads(json_part)
                valid_data = []
                all_valid = True
                for item in extracted:
                   if isinstance(item, dict) and 'localisation' in item and 'nombre' in item:
                        try: item['nombre'] = int(item['nombre'])
                        except (ValueError, TypeError): item['nombre'] = 0; all_valid = False
                        if 'nombre_seances' in item:
                            try: item['nombre_seances'] = int(item['nombre_seances'])
                            except (ValueError, TypeError):
                                if 'nombre_seances' in item: del item['nombre_seances']
                        valid_data.append(item)
                   else:
                        all_valid = False
                if not all_valid:
                     st.warning("Le JSON extrait manuellement n'a pas le bon format pour tous les éléments.")
                return valid_data, raw_response
            except Exception:
                st.error("Impossible d'interpréter la réponse de l'IA.")
                return [], raw_response
    except openai.APIError as e:
        st.error(f"Erreur OpenAI : {e}")
        return [], raw_response
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
        return [], raw_response

def geo_localisation(adresse: str):
    """
    Tente de trouver les coordonnées (latitude, longitude) pour une adresse donnée
    en utilisant Nominatim. Affiche les erreurs/warnings directement dans Streamlit.
    Retourne un tuple (lat, lon) ou None si introuvable ou en cas d'erreur.
    """
    corrections = {
        "région parisienne": "Paris, France", "idf": "Paris, France", "île-de-france": "Paris, France", "ile de france": "Paris, France",
        "sud": "Marseille, France", "le sud": "Marseille, France", "paca": "Marseille, France", "provence-alpes-côte d'azur": "Marseille, France",
        "nord": "Lille, France", "le nord": "Lille, France", "hauts-de-france": "Lille, France",
        "bretagne": "Rennes, France", "côte d'azur": "Nice, France",
        "rhône-alpes": "Lyon, France", "auvergne-rhône-alpes": "Lyon, France",
        "aquitaine": "Bordeaux, France", "nouvelle-aquitaine": "Bordeaux, France",
        "alsace": "Strasbourg, France", "grand est": "Strasbourg, France",
        "france": "Paris, France", "territoire français": "Paris, France",
        "ouest": "Nantes, France", "normandie": "Rouen, France",
        "centre": "Orléans, France", "centre-val de loire": "Orléans, France",
        "auvergne": "Clermont-Ferrand, France"
    }
    adresse_norm = adresse.lower().strip()
    adresse_corrigee = corrections.get(adresse_norm, adresse)
    if ", france" not in adresse_corrigee.lower():
        adresse_requete = f"{adresse_corrigee}, France"
    else:
        adresse_requete = adresse_corrigee
    try:
        loc = geolocator.geocode(adresse_requete)
        if loc:
            return (loc.latitude, loc.longitude)
        else:
            st.warning(f"⚠️ Adresse '{adresse_requete}' (issue de '{adresse}') non trouvée par le service de géolocalisation.")
            return None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        st.error(f"❌ Erreur de géocodage (timeout/indisponible) pour '{adresse_requete}': {e}")
        return None
    except Exception as e:
        st.error(f"❌ Erreur inattendue lors du géocodage de '{adresse_requete}': {e}")
        return None

def trouver_cinemas_proches(localisation_cible: str, spectateurs_voulus: int, nombre_de_salles_voulues: int, rayon_km: int = 50):
    """
    Trouve des cinémas proches d'une localisation cible, pour un nombre EXACT de salles.
    Affiche les warnings/infos directement dans Streamlit.
    Retourne list: Liste des salles sélectionnées.
    """
    point_central_coords = geo_localisation(localisation_cible)
    if not point_central_coords:
        return []

    salles_eligibles = []
    for cinema in cinemas_data:
        lat, lon = cinema.get('lat'), cinema.get('lon')
        if lat is None or lon is None: continue
        try:
            distance = geodesic(point_central_coords, (lat, lon)).km
        except Exception as e:
            st.warning(f"⚠️ Erreur calcul distance pour {cinema.get('cinema', 'Inconnu')} : {e}")
            continue
        if distance > rayon_km: continue
        salles = cinema.get("salles", [])
        # Ne garder que les 2 meilleures salles (par capacité décroissante)
        # Nettoyage : on filtre les salles avec une capacité convertible en int
        salles_valides = []
        for s in salles:
            try:
                capacite = int(s.get("capacite", 0))
                if capacite > 0:
                    s["capacite"] = capacite
                    salles_valides.append(s)
            except (ValueError, TypeError):
                continue

        # Tri et limitation à 2 salles max par cinéma
        salles = sorted(salles_valides, key=lambda s: s["capacite"], reverse=True)[:1]
        for salle in salles:
            try: capacite = int(salle.get("capacite", 0))
            except (ValueError, TypeError): continue
            if capacite <= 0: continue # Ignore salles capacité nulle
            salles_eligibles.append({
                "cinema": cinema.get("cinema"), "salle": salle.get("salle"),
                "adresse": cinema.get("adresse"), "lat": lat, "lon": lon,
                "capacite": capacite, "distance_km": round(distance, 2),
                "contact": cinema.get("contact", {}),
                "source_localisation": localisation_cible
            })

    if not salles_eligibles:
        st.warning(f"Aucune salle trouvée pour '{localisation_cible}' dans un rayon de {rayon_km} km.")
        return []

    salles_eligibles.sort(key=lambda x: (x["distance_km"], -x["capacite"]))

    if len(salles_eligibles) < nombre_de_salles_voulues:
         st.warning(f"⚠️ Seulement {len(salles_eligibles)} salle(s) trouvée(s) pour '{localisation_cible}' (au lieu de {nombre_de_salles_voulues} demandées).")
         resultats = salles_eligibles
    else:
        resultats = salles_eligibles[:nombre_de_salles_voulues]

    return resultats

def generer_carte_folium(groupes_de_cinemas: list):
    """
    Crée une carte Folium affichant les cinémas trouvés, regroupés par couleur.
    Retourne folium.Map or None.
    """
    tous_les_cinemas = [cinema for groupe in groupes_de_cinemas for cinema in groupe.get("resultats", [])]
    if not tous_les_cinemas: return None

    avg_lat = sum(c['lat'] for c in tous_les_cinemas) / len(tous_les_cinemas)
    avg_lon = sum(c['lon'] for c in tous_les_cinemas) / len(tous_les_cinemas)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6, tiles="CartoDB positron")
    couleurs = ["blue", "green", "red", "purple", "orange", "darkred", "lightred", "beige", "darkblue", "darkgreen", "cadetblue", "lightgray", "black"]

    for idx, groupe in enumerate(groupes_de_cinemas):
        couleur = couleurs[idx % len(couleurs)]
        localisation_origine = groupe.get("localisation", "Inconnue")
        resultats_groupe = groupe.get("resultats", [])
        if resultats_groupe:
            feature_group = folium.FeatureGroup(name=f"{localisation_origine} ({len(resultats_groupe)} salles)")
            for cinema in resultats_groupe:
                contact = cinema.get("contact", {})
                contact_nom, contact_email = contact.get("nom", "N/A"), contact.get("email", "N/A")
                cinema["contact_nom"], cinema["contact_email"] = contact_nom, contact_email # Pour table
                popup_html = (f"<b>{cinema.get('cinema', 'N/A')} - Salle {cinema.get('salle', 'N/A')}</b><br>"
                              f"<i>{cinema.get('adresse', 'N/A')}</i><br>"
                              f"Capacité : {cinema.get('capacite', 'N/A')} places<br>"
                              f"Distance ({localisation_origine}) : {cinema.get('distance_km', 'N/A')} km<br>"
                              f"Contact : <b>{contact_nom}</b><br>📧 {contact_email}")
                folium.CircleMarker(
                    location=[cinema['lat'], cinema['lon']], radius=5, color=couleur,
                    fill=True, fill_color=couleur, fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(feature_group)
            feature_group.add_to(m)
    folium.LayerControl().add_to(m)
    return m

def analyser_contexte_geographique(description_projet: str):
    """
    Analyse le contexte du projet pour suggérer les régions les plus pertinentes
    en fonction du public cible, du thème du film, etc.
    Retourne un dictionnaire avec les régions suggérées et leur justification.
    """
    system_prompt = (
        "Tu es un expert en distribution cinématographique et en analyse démographique en France.\n\n"
        "🎯 Ton objectif : analyser le contexte d'un projet cinématographique pour suggérer les régions les plus pertinentes.\n\n"
        "Considère les facteurs suivants :\n"
        "1. Public cible (âge, centres d'intérêt)\n"
        "2. Thème du film\n"
        "3. Type d'événement (avant-première, test, etc.)\n"
        "4. Contexte local (activités, industries, centres d'intérêt)\n\n"
        "Retourne un JSON avec :\n"
        "- regions : liste des régions suggérées\n"
        "- justification : explication pour chaque région\n"
        "- public_cible : description du public cible identifié\n"
        "- facteurs_cles : liste des facteurs qui ont influencé le choix\n\n"
        "Exemple de format de réponse :\n"
        "{\n"
        '  "regions": ["Île-de-France", "Lyon", "Bordeaux"],\n'
        '  "justification": "Ces régions ont une forte concentration de jeunes urbains et d\'activités liées au thème",\n'
        '  "public_cible": "Jeunes adultes 18-35 ans, urbains, intéressés par le thème",\n'
        '  "facteurs_cles": ["Population jeune", "Centres urbains", "Activités liées au thème"]\n'
        "}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description_projet}
            ]
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"Erreur lors de l'analyse du contexte : {e}")
        return None

# --- Fonctions utilitaires ---
def normaliser_nom_ville(nom):
    """
    Normalise un nom de ville/zone pour comparaison : enlève accents, met en minuscules, remplace tirets/underscores par espaces, supprime espaces multiples.
    """
    nom = ''.join(
        c for c in unicodedata.normalize('NFD', nom)
        if unicodedata.category(c) != 'Mn'
    )
    nom = nom.lower()
    nom = re.sub(r"[-_]", " ", nom)
    nom = re.sub(r"\s+", " ", nom)
    nom = nom.strip()
    return nom

# --- Interface Utilisateur Streamlit ---
st.title("🗺️ Assistant de Planification Cinéma MK2")
st.markdown("Décrivez votre projet de diffusion et l'IA identifiera les cinémas pertinents en France.")

if cinemas_ignored_info:
    st.info(f"ℹ️ {cinemas_ignored_info}")

with st.expander("ℹ️ Comment ça marche ?"):
    st.markdown("""
    Cette application vous aide à planifier des projections de films en identifiant les cinémas les plus adaptés en France.
    ### 📝 1. Décrivez votre projet
    Indiquez votre projet en détail : thème du film, public cible, type d'événement, etc.
    *Exemples :*
    - "Film sur l'automobile par Inoxtag, public jeune"
    - "Documentaire sur l'agriculture bio, public adulte"
    - "Film d'animation pour enfants"
    ### 🎯 2. Analyse du contexte
    L'IA analyse votre projet pour suggérer les régions les plus pertinentes en fonction du public cible et du thème.
    ### 🤖 3. Planification détaillée
    Précisez ensuite votre besoin en langage naturel : lieux, type d'événement et public cible.
    ### 🔍 4. Recherche de cinémas
    Le système cherche les salles adaptées dans les régions suggérées.
    ### 🗺️ 5. Carte interactive
    Une carte Folium affiche les cinémas trouvés.
    ### 📊 6. Liste des Salles et Export
    - Tableau récapitulatif des salles
    - Export Excel disponible
    """)

# Première étape : Analyse du contexte
st.subheader("🎯 Analyse du Contexte")
description_projet = st.text_area(
    "Décrivez votre projet :",
    placeholder="Ex: Film sur l'automobile par Inoxtag, public jeune, avant-première",
    key="description_projet"
)

# Bouton pour déclencher l'analyse du contexte
if st.button("🔍 Analyser le contexte", type="primary"):
    if description_projet:
        with st.spinner("🧠 Analyse du contexte par l'IA..."):
            contexte = analyser_contexte_geographique(description_projet)
            st.session_state.contexte_result = contexte
            st.session_state.analyse_contexte_done = True
        st.rerun()
    else:
        st.warning("Veuillez d'abord décrire votre projet.")

# Affichage des résultats de l'analyse du contexte
if st.session_state.analyse_contexte_done and st.session_state.contexte_result:
    st.success("✅ Analyse du contexte terminée !")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Public cible identifié :**")
        st.info(st.session_state.contexte_result.get("public_cible", "Non spécifié"))
        
        st.markdown("**🎯 Facteurs clés :**")
        for facteur in st.session_state.contexte_result.get("facteurs_cles", []):
            st.markdown(f"- {facteur}")
    
    with col2:
        st.markdown("**🗺️ Régions suggérées :**")
        for region in st.session_state.contexte_result.get("regions", []):
            st.markdown(f"- {region}")
        
        st.markdown("**💡 Justification :**")
        st.info(st.session_state.contexte_result.get("justification", "Non spécifié"))
    
    st.markdown("---")
    st.subheader("📝 Planification détaillée")
    st.info("Maintenant que nous avons identifié les régions pertinentes, détaillez votre plan de diffusion.")

# Deuxième étape : Planification détaillée
query = st.text_input(
    "Votre plan de diffusion :",
    placeholder="Ex: 5 séances à Paris (500 pers.) et 2 séances test à Rennes (100 pers.)",
    key="query_input"
)

# Bouton pour déclencher l'analyse de la requête
if st.button("🤖 Analyser la requête", type="primary"):
    if query:
        with st.spinner("🧠 Interprétation de votre requête par l'IA..."):
            instructions_ia, reponse_brute_ia = analyser_requete_ia(query)
            st.session_state.instructions_ia = instructions_ia
            st.session_state.reponse_brute_ia = reponse_brute_ia
        st.rerun()
    else:
        st.warning("Veuillez d'abord saisir votre plan de diffusion.")

# Affichage des résultats de l'analyse de la requête
if st.session_state.instructions_ia:
    total_spectateurs_estimes = sum(i.get('nombre', 0) for i in st.session_state.instructions_ia)
    total_seances_demandees_ia = sum(i.get("nombre_seances", 0) for i in st.session_state.instructions_ia if "nombre_seances" in i)
    nb_zones = len(st.session_state.instructions_ia)

    # Modifié ici : expanded=False pour que l'expander soit fermé par défaut
    with st.expander("🤖 Résumé de la compréhension de l'IA", expanded=False):
        resume_text = f"**IA a compris :** {nb_zones} zone(s) de recherche"
        if total_spectateurs_estimes > 0: resume_text += f" pour un objectif total d'environ {total_spectateurs_estimes} spectateurs"
        if total_seances_demandees_ia > 0: resume_text += f" et un total de {total_seances_demandees_ia} séance(s) explicitement demandée(s)."
        else: resume_text += "."; st.caption("(Aucun nombre de séances spécifique n'a été détecté, une estimation sera faite.)")
        st.info(resume_text)
        st.json(st.session_state.instructions_ia)
        if st.session_state.reponse_brute_ia:
            with st.popover("Voir réponse brute de l'IA"):
                st.code(st.session_state.reponse_brute_ia, language="text")

    # Configuration des rayons de recherche
    st.sidebar.header("⚙️ Options de Recherche")
    rayons_par_loc = {}
    for idx, instruction in enumerate(st.session_state.instructions_ia):
        loc = instruction.get('localisation')
        if loc:
             corrections_regionales = ["paris", "lille", "marseille", "toulouse", "nice", "nantes", "rennes", "strasbourg", "clermont-ferrand", "lyon", "bordeaux", "rouen", "orléans"]
             is_large_area_target = loc.lower() in ["marseille", "toulouse", "nice", "lille", "nantes", "rennes", "strasbourg", "clermont-ferrand", "lyon", "bordeaux"] or loc.lower() in ["paris"] and len(st.session_state.instructions_ia) > 1
             default_rayon = 100 if is_large_area_target else 50
             rayon_key = f"rayon_{idx}_{loc}"
             if is_large_area_target: st.sidebar.caption(f"'{loc}' peut couvrir une zone large, rayon par défaut ajusté.")
             rayons_par_loc[loc] = st.sidebar.slider(f"Rayon autour de '{loc}' (km)", 5, 250, default_rayon, 5, key=rayon_key)

    # Bouton pour déclencher la recherche des cinémas
    if st.button("🔍 Rechercher les cinémas", type="primary"):
        liste_groupes_resultats = []
        cinemas_trouves_total = 0
        total_seances_estimees_ou_demandees = 0
        dataframes_to_export = {}
        
        st.markdown("---")
        st.subheader("🔍 Recherche des cinémas...")
        
        with st.spinner(f"Recherche en cours pour {nb_zones} zone(s)..."):
            for instruction in st.session_state.instructions_ia:
                loc = instruction.get('localisation')
                num_spectateurs = instruction.get('nombre')
                if loc and isinstance(num_spectateurs, int) and num_spectateurs >= 0:
                    st.write(f"**Recherche pour : {loc}**")
                    rayon_recherche = rayons_par_loc.get(loc, 50)
                    if "nombre_seances" in instruction and isinstance(instruction["nombre_seances"], int) and instruction["nombre_seances"] > 0:
                        nombre_salles_a_trouver = instruction["nombre_seances"]
                        st.info(f"   -> Objectif : trouver {nombre_salles_a_trouver} salle(s) dans {rayon_recherche} km (cible: {num_spectateurs} spect.).")
                    else:
                        nombre_salles_a_trouver = 1
                        st.info(f"   -> Objectif : trouver {nombre_salles_a_trouver} salle (défaut) dans {rayon_recherche} km (cible: {num_spectateurs} spect.).")
                    total_seances_estimees_ou_demandees += nombre_salles_a_trouver
                    resultats_cinemas = trouver_cinemas_proches(loc, num_spectateurs, nombre_salles_a_trouver, rayon_recherche)
                    groupe_actuel = {"localisation": loc, "resultats": resultats_cinemas, "nombre_salles_demandees": nombre_salles_a_trouver}
                    liste_groupes_resultats.append(groupe_actuel)
                    if resultats_cinemas:
                        capacite_trouvee = sum(c['capacite'] for c in resultats_cinemas)
                        st.write(f"   -> Trouvé {len(resultats_cinemas)} salle(s) (Capacité totale: {capacite_trouvee}).")
                        cinemas_trouves_total += len(resultats_cinemas)
                        data_for_df = []
                        for cinema in resultats_cinemas:
                            contact = cinema.get("contact", {})
                            data_for_df.append({
                                "Cinéma": cinema.get("cinema", "N/A"), 
                                "Salle": cinema.get("salle", "N/A"),
                                "Adresse": cinema.get("adresse", "N/A"), 
                                "Capacité": cinema.get("capacite", 0),
                                "Distance (km)": cinema.get("distance_km", 0), 
                                # Remplacer les deux colonnes par une seule avec les informations combinées
                                "Contact": " / ".join(filter(None, [
                                    contact.get("nom", ""), 
                                    contact.get("email", ""), 
                                    contact.get("telephone", "")
                                ])),
                                "Latitude": cinema.get("lat", 0.0),
                                "Longitude": cinema.get("lon", 0.0) 
                            })
                        df = pd.DataFrame(data_for_df)
                        if not df.empty: dataframes_to_export[loc] = df
                    else: st.write(f"   -> Aucune salle trouvée pour '{loc}' correspondant aux critères.")
                else: st.warning(f"Instruction IA ignorée (format invalide) : {instruction}")
        
        # Sauvegarde des résultats dans la session
        st.session_state.liste_groupes_resultats = liste_groupes_resultats
        st.session_state.dataframes_to_export = dataframes_to_export
        st.session_state.recherche_cinemas_done = True
        st.session_state.modifications_appliquees = False  # Réinitialiser les modifications
        st.rerun()

# Affichage des résultats de la recherche
if st.session_state.recherche_cinemas_done and st.session_state.liste_groupes_resultats:
    st.markdown("---")
    st.subheader("📊 Résultats de la Recherche")
    
    total_seances_estimees_ou_demandees = sum(groupe.get("nombre_salles_demandees", 0) for groupe in st.session_state.liste_groupes_resultats)
    cinemas_trouves_total = sum(len(groupe.get("resultats", [])) for groupe in st.session_state.liste_groupes_resultats)
    salles_manquantes = total_seances_estimees_ou_demandees - cinemas_trouves_total
    
    if cinemas_trouves_total > 0:
        if salles_manquantes > 0: st.warning(f"⚠️ Recherche terminée. {cinemas_trouves_total} salle(s) trouvée(s), mais il en manque {salles_manquantes} sur les {total_seances_estimees_ou_demandees} visée(s).")
        else: st.success(f"✅ Recherche terminée ! {cinemas_trouves_total} salle(s) trouvée(s), correspondant aux {total_seances_estimees_ou_demandees} séance(s) visée(s).")

        st.subheader("🗺️ Carte des Cinémas Trouvés")
        carte = generer_carte_folium(st.session_state.liste_groupes_resultats)
        if carte:
            map_html_path = "map_output.html"
            carte.save(map_html_path)
            st_folium(carte, width='100%', height=500, key="carte_principale")
            with open(map_html_path, "rb") as f:
                st.download_button("📥 Télécharger la Carte Interactive (HTML)", f, "carte_cinemas.html", "text/html", use_container_width=True, key="download_carte_principale")
            with st.expander("💡 Comment utiliser le fichier HTML ?"):
                  st.markdown("- Double-cliquez sur `carte_cinemas.html`.\n- S'ouvre dans votre navigateur.\n- Carte interactive: zoom, déplacement, clic sur points.\n- Contrôle des couches pour filtrer par zone.\n- Fonctionne hors ligne.")
        else: st.info("Génération de la carte annulée.")

        st.markdown("---")
        st.subheader("📋 Liste des Salles et Export")

        if st.session_state.dataframes_to_export:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                for loc, df_to_write in st.session_state.dataframes_to_export.items():
                    safe_sheet_name = "".join(c for c in loc if c.isalnum() or c in (' ', '_')).rstrip()[:31]
                    df_to_write.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            st.download_button(
                label="💾 Télécharger Tous les Résultats (Excel)",
                data=excel_buffer.getvalue(),
                file_name=f"resultats_cinemas_{uuid.uuid4()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="download_all_excel" )

        for groupe in st.session_state.liste_groupes_resultats:
            loc = groupe["localisation"]
            nb_demandes = groupe["nombre_salles_demandees"]
            nb_trouves = len(groupe["resultats"])
            st.markdown(f"**Zone : {loc}** ({nb_trouves}/{nb_demandes} salles trouvées)")
            if loc in st.session_state.dataframes_to_export:
                df_display = st.session_state.dataframes_to_export[loc]
                st.dataframe(df_display[["Cinéma", "Salle", "Capacité", "Distance (km)", "Contact"]], use_container_width=True, hide_index=True)
            elif nb_trouves == 0 : st.caption("Aucune salle trouvée pour cette zone.")
            st.divider()
    else:
         st.error("❌ Aucun cinéma correspondant à votre demande n'a été trouvé.")
         if salles_manquantes > 0: st.info(f"(L'objectif était de trouver {total_seances_estimees_ou_demandees} salle(s).)")

# --- Fin de l'application ---

# Section de raffinage des résultats
if st.session_state.recherche_cinemas_done and st.session_state.liste_groupes_resultats:
    st.markdown("---")
    st.subheader("🔧 Raffinage des Résultats")
    st.info("Vous pouvez maintenant affiner vos résultats en demandant des modifications spécifiques.")
    
    # Chatbox pour le raffinage
    raffinage_query = st.text_input(
        "Demande de modification :",
        placeholder="Ex: rajoute une salle à Marseille, supprime les salles de moins de 100 places, ajoute 2 salles à Paris",
        key="raffinage_input"
    )
    
    if st.button("🔧 Appliquer les modifications", type="secondary"):
        if raffinage_query:
            st.write(f"🔍 **DEBUG :** Demande de raffinage reçue : '{raffinage_query}'")
            st.write(f"🔍 **DEBUG :** État initial - Groupes : {len(st.session_state.liste_groupes_resultats)}")
            for i, groupe in enumerate(st.session_state.liste_groupes_resultats):
                st.write(f"🔍 **DEBUG :** Groupe {i+1} : {groupe['localisation']} - {len(groupe['resultats'])} salles")
            with st.spinner("🧠 Traitement de votre demande de modification..."):
                # Analyse de la demande de raffinage
                system_prompt_raffinage = (
                    "Tu es un expert en analyse de requêtes de modification pour une application de planification cinématographique.\n\n"
                    "🎯 Ton objectif : analyser une demande de modification et retourner des instructions claires.\n\n"
                    "Types de modifications supportées :\n"
                    "1. AJOUTER des salles ou séances : 'rajoute X salles à [ville]', 'ajoute 10 séances à [ville]', 'ajoute une salle à [ville]'\n"
                    "2. SUPPRIMER des salles ou séances : 'supprime les salles de moins de X places', 'enlève les salles à plus de X km', 'supprime les séances à [ville]'\n"
                    "3. MODIFIER des critères : 'augmente le rayon à X km pour [ville]', 'cherche des salles de plus de X places'\n\n"
                    "Le terme 'séance(s)' doit être compris comme 'salle(s)' dans ce contexte.\n"
                    "Exemples :\n"
                    "- 'ajoute 10 séances à Paris' => ajouter 10 salles à Paris\n"
                    "- 'supprime les séances à Marseille' => supprimer toutes les salles à Marseille\n"
                    "- 'supprime les salles à Lyon' => supprimer toutes les salles à Lyon\n\n"
                    "Retourne un JSON avec :\n"
                    "- action : 'ajouter', 'supprimer', 'modifier'\n"
                    "- localisation : ville concernée (si applicable)\n"
                    "- nombre : nombre de salles (pour ajout)\n"
                    "- critere : critère de suppression/modification ('capacite_min', 'capacite_max', 'distance_max')\n"
                    "- valeur : valeur du critère (nombre)\n"
                    "- operateur : 'superieur', 'inferieur', 'egal' (pour clarifier la logique)\n\n"
                    "Exemple :\n"
                    "{\n"
                    '  "action": "ajouter",\n'
                    '  "localisation": "Marseille",\n'
                    '  "nombre": 1\n'
                    "}\n\n"
                    "Pour les suppressions :\n"
                    "{\n"
                    '  "action": "supprimer",\n'
                    '  "critere": "capacite_min",\n'
                    '  "valeur": 100,\n'
                    '  "operateur": "inferieur"\n'
                    "}\n\n"
                    "Pour supprimer toutes les salles d'une ville :\n"
                    "{\n"
                    '  "action": "supprimer",\n'
                    '  "localisation": "Marseille"\n'
                    "}\n\n"
                    "Si la demande n'est pas claire ou non supportée, retourne :\n"
                    "{\n"
                    '  "action": "incompris",\n'
                    '  "message": "explication"\n'
                    "}\n\n"
                    "⚠️ IMPORTANT : Retourne UNIQUEMENT le JSON, sans préfixe 'json', sans backticks, sans texte avant ou après."
                )
                
                try:
                    st.write("🔍 **DEBUG :** Envoi de la demande à l'IA...")
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt_raffinage},
                            {"role": "user", "content": raffinage_query}
                        ]
                    )
                    raw_response = response.choices[0].message.content.strip()
                    st.write(f"🔍 **DEBUG :** Réponse brute de l'IA : {raw_response}")
                    
                    # Nettoyer la réponse pour enlever les préfixes comme "json "
                    cleaned_response = raw_response
                    if raw_response.startswith("json "):
                        cleaned_response = raw_response[5:]  # Enlever "json "
                    elif raw_response.startswith("```json"):
                        cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()
                    elif raw_response.startswith("```"):
                        cleaned_response = raw_response.replace("```", "").strip()
                    
                    st.write(f"🔍 **DEBUG :** Réponse nettoyée : {cleaned_response}")
                    raffinage_instruction = json.loads(cleaned_response)
                    st.write(f"🔍 **DEBUG :** Instruction parsée : {raffinage_instruction}")
                    
                    # --- Normalisation des instructions IA (séance(s) -> salle(s)) ---
                    def normaliser_instruction_ia(instr):
                        # Si l'action ou les champs contiennent 'séance', on les remplace par 'salle'
                        if 'action' in instr and isinstance(instr['action'], str):
                            instr['action'] = instr['action'].replace('séance', 'salle').replace('séances', 'salles')
                        if 'critere' in instr and isinstance(instr['critere'], str):
                            instr['critere'] = instr['critere'].replace('séance', 'salle').replace('séances', 'salles')
                        # Si le champ nombre est présent, s'assurer que c'est un int
                        if 'nombre' in instr:
                            try:
                                instr['nombre'] = int(instr['nombre'])
                            except Exception:
                                instr['nombre'] = 1
                        return instr
                    raffinage_instruction = normaliser_instruction_ia(raffinage_instruction)
                    
                    # Application des modifications
                    modifications_appliquees = False
                    st.write(f"🔍 **DEBUG :** Action détectée : {raffinage_instruction.get('action')}")
                    
                    if raffinage_instruction.get("action") == "ajouter":
                        localisation = raffinage_instruction.get("localisation")
                        nombre = raffinage_instruction.get("nombre", 1)
                        st.write(f"🔍 **DEBUG :** Ajout - Localisation : '{localisation}', Nombre : {nombre}")
                        
                        # Validation des données
                        validation_ok = True
                        if not localisation:
                            st.error("❌ Localisation manquante dans la demande d'ajout")
                            validation_ok = False
                        
                        if validation_ok:
                            try:
                                nombre = int(nombre)
                                if nombre <= 0:
                                    st.error("❌ Le nombre de salles doit être positif")
                                    validation_ok = False
                            except (ValueError, TypeError):
                                st.error("❌ Nombre de salles invalide")
                                validation_ok = False
                        
                        if validation_ok:
                            st.write(f"🔍 **DEBUG :** Validation OK, recherche du groupe existant...")
                            # Trouver le groupe existant ou en créer un nouveau
                            groupe_existant = None
                            for groupe in st.session_state.liste_groupes_resultats:
                                st.write(f"🔍 **DEBUG :** Comparaison : '{normaliser_nom_ville(groupe['localisation'])}' vs '{normaliser_nom_ville(localisation)}'")
                                if normaliser_nom_ville(groupe["localisation"]) == normaliser_nom_ville(localisation):
                                    groupe_existant = groupe
                                    st.write(f"🔍 **DEBUG :** Groupe existant trouvé pour {localisation}")
                                    break
                            
                            if groupe_existant:
                                st.write(f"🔍 **DEBUG :** Ajout au groupe existant pour {localisation}")
                                # Ajouter des salles au groupe existant
                                rayon_actuel = 100  # rayon augmenté pour plus de flexibilité
                                st.write(f"🔍 **DEBUG :** Recherche de {nombre * 2} salles supplémentaires dans un rayon de {rayon_actuel} km")
                                resultats_supplementaires = trouver_cinemas_proches(
                                    localisation, 
                                    1000,  # objectif spectateurs par défaut
                                    nombre * 2,  # chercher plus pour avoir du choix
                                    rayon_actuel
                                )
                                st.write(f"🔍 **DEBUG :** {len(resultats_supplementaires)} salles supplémentaires trouvées")
                                
                                # Filtrer pour éviter les doublons (plus robuste)
                                salles_existantes = set()
                                for c in groupe_existant["resultats"]:
                                    # Créer un identifiant unique basé sur plusieurs critères
                                    identifiant = f"{c['cinema']}_{c['salle']}_{c.get('adresse', '')}"
                                    salles_existantes.add(identifiant.lower())
                                st.write(f"🔍 **DEBUG :** {len(salles_existantes)} salles existantes identifiées")
                                
                                nouvelles_salles = []
                                for s in resultats_supplementaires:
                                    identifiant = f"{s['cinema']}_{s['salle']}_{s.get('adresse', '')}"
                                    if identifiant.lower() not in salles_existantes:
                                        nouvelles_salles.append(s)
                                        st.write(f"🔍 **DEBUG :** Nouvelle salle ajoutée : {s['cinema']} - {s['salle']}")
                                        if len(nouvelles_salles) >= nombre:
                                            break
                                st.write(f"🔍 **DEBUG :** {len(nouvelles_salles)} nouvelles salles uniques trouvées")
                                
                                if nouvelles_salles:
                                    st.write(f"🔍 **DEBUG :** Ajout de {len(nouvelles_salles)} salles au groupe existant")
                                    groupe_existant["resultats"].extend(nouvelles_salles)
                                    groupe_existant["nombre_salles_demandees"] += len(nouvelles_salles)
                                    modifications_appliquees = True
                                    st.success(f"✅ {len(nouvelles_salles)} nouvelle(s) salle(s) ajoutée(s) à {localisation}")
                                    
                                    # Mettre à jour le dataframe d'export
                                    if localisation in st.session_state.dataframes_to_export:
                                        df_existant = st.session_state.dataframes_to_export[localisation]
                                        for salle in nouvelles_salles:
                                            contact = salle.get("contact", {})
                                            nouvelle_ligne = {
                                                "Cinéma": salle.get("cinema", "N/A"),
                                                "Salle": salle.get("salle", "N/A"),
                                                "Adresse": salle.get("adresse", "N/A"),
                                                "Capacité": salle.get("capacite", 0),
                                                "Distance (km)": salle.get("distance_km", 0),
                                                "Contact": " / ".join(filter(None, [
                                                    contact.get("nom", ""),
                                                    contact.get("email", ""),
                                                    contact.get("telephone", "")
                                                ])),
                                                "Latitude": salle.get("lat", 0.0),
                                                "Longitude": salle.get("lon", 0.0)
                                            }
                                            df_existant = pd.concat([df_existant, pd.DataFrame([nouvelle_ligne])], ignore_index=True)
                                        st.session_state.dataframes_to_export[localisation] = df_existant
                                else:
                                    st.warning(f"⚠️ Aucune nouvelle salle trouvée pour {localisation}")
                            else:
                                st.write(f"🔍 **DEBUG :** Création d'un nouveau groupe pour {localisation}")
                                # Créer un nouveau groupe
                                resultats_nouveaux = trouver_cinemas_proches(
                                    localisation,
                                    1000,
                                    nombre,
                                    100  # rayon plus large pour les nouveaux groupes
                                )
                                st.write(f"🔍 **DEBUG :** {len(resultats_nouveaux)} salles trouvées pour le nouveau groupe")
                                if resultats_nouveaux:
                                    nouveau_groupe = {
                                        "localisation": localisation,
                                        "resultats": resultats_nouveaux,
                                        "nombre_salles_demandees": len(resultats_nouveaux)
                                    }
                                    st.session_state.liste_groupes_resultats.append(nouveau_groupe)
                                    modifications_appliquees = True
                                    st.success(f"✅ Nouveau groupe créé pour {localisation} avec {len(resultats_nouveaux)} salle(s)")
                                    
                                    # Créer le dataframe d'export
                                    data_for_df = []
                                    for cinema in resultats_nouveaux:
                                        contact = cinema.get("contact", {})
                                        data_for_df.append({
                                            "Cinéma": cinema.get("cinema", "N/A"),
                                            "Salle": cinema.get("salle", "N/A"),
                                            "Adresse": cinema.get("adresse", "N/A"),
                                            "Capacité": cinema.get("capacite", 0),
                                            "Distance (km)": cinema.get("distance_km", 0),
                                            "Contact": " / ".join(filter(None, [
                                                contact.get("nom", ""),
                                                contact.get("email", ""),
                                                contact.get("telephone", "")
                                            ])),
                                            "Latitude": cinema.get("lat", 0.0),
                                            "Longitude": cinema.get("lon", 0.0)
                                        })
                                    st.session_state.dataframes_to_export[localisation] = pd.DataFrame(data_for_df)
                                else:
                                    st.warning(f"⚠️ Aucune salle trouvée pour {localisation}")
                    
                    elif raffinage_instruction.get("action") == "supprimer":
                        critere = raffinage_instruction.get("critere")
                        valeur = raffinage_instruction.get("valeur")
                        operateur = raffinage_instruction.get("operateur", "inferieur")
                        localisation = raffinage_instruction.get("localisation")
                        st.write(f"🔍 **DEBUG :** Suppression - Critère : '{critere}', Valeur : {valeur}, Opérateur : '{operateur}', Localisation : '{localisation}'")

                        # Cas spécial : suppression de toutes les salles d'une localisation
                        if localisation and not critere:
                            groupes_supprimes = 0
                            for groupe in st.session_state.liste_groupes_resultats:
                                if groupe["localisation"].lower() == localisation.lower():
                                    nb_salles = len(groupe["resultats"])
                                    groupe["resultats"] = []
                                    groupes_supprimes += nb_salles
                                    st.write(f"🔍 **DEBUG :** Toutes les salles supprimées pour {localisation} ({nb_salles} salles)")
                                    # Mettre à jour le dataframe d'export
                                    if localisation in st.session_state.dataframes_to_export:
                                        st.session_state.dataframes_to_export[localisation] = st.session_state.dataframes_to_export[localisation].iloc[0:0]
                            if groupes_supprimes > 0:
                                modifications_appliquees = True
                                st.success(f"✅ Toutes les salles supprimées pour {localisation} ({groupes_supprimes} salles)")
                            else:
                                st.info(f"ℹ️ Aucune salle à supprimer pour {localisation}")
                        else:
                            # Validation des données
                            validation_ok = True
                            if not critere or valeur is None:
                                st.error("❌ Critère ou valeur manquant pour la suppression")
                                validation_ok = False
                            if validation_ok:
                                try:
                                    valeur = float(valeur)
                                except (ValueError, TypeError):
                                    st.error("❌ Valeur numérique invalide pour le critère")
                                    validation_ok = False
                            if validation_ok and critere and valeur is not None:
                                st.write(f"🔍 **DEBUG :** Validation OK, début de la suppression...")
                                salles_supprimees = 0
                                for groupe in st.session_state.liste_groupes_resultats:
                                    resultats_originaux = groupe["resultats"].copy()
                                    st.write(f"🔍 **DEBUG :** Traitement du groupe {groupe['localisation']} : {len(resultats_originaux)} salles avant filtrage")
                                    # Logique de filtrage améliorée avec opérateurs
                                    if critere == "capacite_min":
                                        if operateur == "inferieur":
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("capacite", 0) >= valeur]
                                        elif operateur == "superieur":
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("capacite", 0) <= valeur]
                                        else:  # egal
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("capacite", 0) == valeur]
                                    elif critere == "capacite_max":
                                        if operateur == "inferieur":
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("capacite", 0) <= valeur]
                                        elif operateur == "superieur":
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("capacite", 0) >= valeur]
                                        else:  # egal
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("capacite", 0) == valeur]
                                    elif critere == "distance_max":
                                        if operateur == "inferieur":
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("distance_km", 0) <= valeur]
                                        elif operateur == "superieur":
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("distance_km", 0) >= valeur]
                                        else:  # egal
                                            groupe["resultats"] = [s for s in resultats_originaux if s.get("distance_km", 0) == valeur]
                                    salles_supprimees_groupe = len(resultats_originaux) - len(groupe["resultats"])
                                    salles_supprimees += salles_supprimees_groupe
                                    st.write(f"🔍 **DEBUG :** Groupe {groupe['localisation']} : {salles_supprimees_groupe} salles supprimées, {len(groupe['resultats'])} restantes")
                                    # Mettre à jour le dataframe d'export
                                    if groupe["localisation"] in st.session_state.dataframes_to_export:
                                        # Recréer le dataframe avec les nouvelles données
                                        data_for_df = []
                                        for cinema in groupe["resultats"]:
                                            contact = cinema.get("contact", {})
                                            data_for_df.append({
                                                "Cinéma": cinema.get("cinema", "N/A"),
                                                "Salle": cinema.get("salle", "N/A"),
                                                "Adresse": cinema.get("adresse", "N/A"),
                                                "Capacité": cinema.get("capacite", 0),
                                                "Distance (km)": cinema.get("distance_km", 0),
                                                "Contact": " / ".join(filter(None, [
                                                    contact.get("nom", ""),
                                                    contact.get("email", ""),
                                                    contact.get("telephone", "")
                                                ])),
                                                "Latitude": cinema.get("lat", 0.0),
                                                "Longitude": cinema.get("lon", 0.0)
                                            })
                                        st.session_state.dataframes_to_export[groupe["localisation"]] = pd.DataFrame(data_for_df)
                                if salles_supprimees > 0:
                                    modifications_appliquees = True
                                    st.success(f"✅ {salles_supprimees} salle(s) supprimée(s) selon le critère : {critere} {operateur} {valeur}")
                                else:
                                    st.info("ℹ️ Aucune salle ne correspondait aux critères de suppression")
                    
                    elif raffinage_instruction.get("action") == "incompris":
                        st.warning(f"⚠️ Demande non comprise : {raffinage_instruction.get('message', 'Format non reconnu')}")
                        st.info("💡 Exemples de demandes supportées :\n- 'rajoute une salle à Marseille'\n- 'supprime les salles de moins de 100 places'\n- 'ajoute 2 salles à Paris'")
                    
                    else:
                        st.warning("⚠️ Type d'action non supporté")
                    
                    # Forcer la mise à jour de l'interface si des modifications ont été appliquées
                    if modifications_appliquees:
                        st.write(f"🔍 **DEBUG :** Modifications appliquées, mise à jour de l'interface...")
                        st.session_state.modifications_appliquees = True
                        st.success("🔄 Interface mise à jour avec les nouvelles données")
                    else:
                        st.write(f"🔍 **DEBUG :** Aucune modification appliquée")
                        
                except json.JSONDecodeError as e:
                    st.error(f"❌ Erreur JSON lors de l'analyse de la demande de modification : {e}")
                    st.write(f"🔍 **DEBUG :** Réponse qui a causé l'erreur : {raw_response}")
                except Exception as e:
                    st.error(f"❌ Erreur inattendue : {e}")
                    st.write(f"🔍 **DEBUG :** Type d'erreur : {type(e).__name__}")
            
            st.rerun()
        else:
            st.warning("Veuillez saisir une demande de modification.")
    
    # Affichage des exemples de raffinage
    with st.expander("💡 Exemples de demandes de raffinage", expanded=False):
        st.markdown("""
        **Ajouter des salles :**
        - "rajoute une salle à Marseille"
        - "ajoute 2 salles à Paris"
        - "ajoute une salle à Lyon"
        
        **Supprimer des salles :**
        - "supprime les salles de moins de 100 places"
        - "enlève les salles à plus de 30 km"
        - "supprime les salles de plus de 200 places"
        
        **Modifier des critères :**
        - "augmente le rayon à 100 km pour Paris"
        - "cherche des salles de plus de 150 places"
        """)
    
    # Affichage des résultats mis à jour après raffinage
    # Cette section ne s'affiche que si des modifications ont été appliquées
    if st.session_state.liste_groupes_resultats and st.session_state.get('modifications_appliquees', False):
        st.markdown("---")
        st.subheader("📊 Résultats Mis à Jour")
        
        total_seances_apres_raffinage = sum(groupe.get("nombre_salles_demandees", 0) for groupe in st.session_state.liste_groupes_resultats)
        cinemas_trouves_apres_raffinage = sum(len(groupe.get("resultats", [])) for groupe in st.session_state.liste_groupes_resultats)
        
        st.info(f"📈 **Total après raffinage :** {cinemas_trouves_apres_raffinage} salle(s) trouvée(s) sur {total_seances_apres_raffinage} séance(s) visée(s)")
        
        # Carte mise à jour
        st.subheader("🗺️ Carte Mise à Jour")
        carte_mise_a_jour = generer_carte_folium(st.session_state.liste_groupes_resultats)
        if carte_mise_a_jour:
            map_html_path = "map_output_raffinage.html"
            carte_mise_a_jour.save(map_html_path)
            st_folium(carte_mise_a_jour, width='100%', height=500, key="carte_raffinage")
            with open(map_html_path, "rb") as f:
                st.download_button("📥 Télécharger la Carte Mise à Jour (HTML)", f, "carte_cinemas_raffinage.html", "text/html", use_container_width=True, key="download_carte_raffinage")
        
        # Tableaux mis à jour
        st.subheader("📋 Tableaux Mis à Jour")
        if st.session_state.dataframes_to_export:
            excel_buffer_raffinage = io.BytesIO()
            with pd.ExcelWriter(excel_buffer_raffinage, engine='xlsxwriter') as writer:
                for loc, df_to_write in st.session_state.dataframes_to_export.items():
                    safe_sheet_name = "".join(c for c in loc if c.isalnum() or c in (' ', '_')).rstrip()[:31]
                    df_to_write.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            st.download_button(
                label="💾 Télécharger Résultats Mis à Jour (Excel)",
                data=excel_buffer_raffinage.getvalue(),
                file_name=f"resultats_cinemas_raffinage_{uuid.uuid4()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="download_raffinage_excel"
            )
        
        # Affichage des tableaux par zone
        for groupe in st.session_state.liste_groupes_resultats:
            loc = groupe["localisation"]
            nb_demandes = groupe["nombre_salles_demandees"]
            nb_trouves = len(groupe["resultats"])
            st.markdown(f"**Zone : {loc}** ({nb_trouves}/{nb_demandes} salles trouvées)")
            if loc in st.session_state.dataframes_to_export:
                df_display = st.session_state.dataframes_to_export[loc]
                st.dataframe(df_display[["Cinéma", "Salle", "Capacité", "Distance (km)", "Contact"]], use_container_width=True, hide_index=True)
            elif nb_trouves == 0:
                st.caption("Aucune salle trouvée pour cette zone.")
            st.divider()

# --- Fin de l'application ---