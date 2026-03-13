import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import time

# -------------------------
# CONFIG
# -------------------------

st.set_page_config(
    page_title="What2Play",
)

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title("🎲 What 2 Play?")
st.sidebar.title("Options")

page = st.sidebar.radio(
    "Go to:",
    ["📖 Recommend a Game", "⭐ Rate a Game"]
)

dark_toggle = st.sidebar.toggle("🌙 Dark Mode")
st.session_state.dark_mode = dark_toggle

st.sidebar.divider()
user = st.sidebar.text_input("User name", "guest")


# -------------------------
# DARK MODE STATE
# -------------------------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# -------------------------
# Session state ratings
# -------------------------

user_file = f"../data/users/{user}_ratings.csv"

if "user_ratings" not in st.session_state or st.session_state.get("current_user") != user:

    try:
        st.session_state.user_ratings = pd.read_csv(user_file)
    except:
        st.session_state.user_ratings = pd.DataFrame(columns=["name","user_rating"])

    st.session_state.current_user = user

# -------------------------
# DARK MODE STYLE
# -------------------------

if st.session_state.dark_mode:

    st.markdown("""
    <style>

    .stApp {
        background-color:#0e1117;
        color:white;
    }

    h1, h2, h3, h4, h5, h6 {
        color:white !important;
    }

    label {
        color:white !important;
    }

    .stSidebar {
        background-color:#0e1117;
    }

    .css-1d391kg {
        color:white;
    }

    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    h1 {
        margin-bottom:60px;
    }

    h2 {
        margin-top:30px;
    }

    </style>
    """, unsafe_allow_html=True)

# -------------------------
# LOAD DATA (cache)
# -------------------------

@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path= BASE_DIR / "data" / "processed" / "boardgames_features.csv"
    return pd.read_csv(data_path)

df = load_data()

# ----------------------
# LOAD MODEL (cache)
# ----------------------

@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "models" / "boardgame_rating_predictor.pkl"
    return joblib.load(model_path)

model = load_model()

# -------------------------
# GAME CARD
# -------------------------

def show_game_card(row):

    st.markdown(f"""
    <div style="
        border-radius:12px;
        padding:18px;
        margin-bottom:18px;
        background-color:#1f2933;
        color:white;
        box-shadow:0px 4px 10px rgba(0,0,0,0.3);
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        min-height:190px;
    ">

    <h4 style="
        color:white;
        margin-bottom:10px;
        font-size:18px;
        line-height:1.2;
        display:-webkit-box;
        -webkit-line-clamp:2;
        -webkit-box-orient:vertical;
        overflow:hidden;
        text-overflow:ellipsis;
    ">
    🎲 {row['name']}
    </h4>

    <div style="font-size:14px; line-height:1.5">

    ⭐ Score: {row['hybrid_score']:.2f} <br>
    👥 Players: {row['minplayers']} - {row['maxplayers']} <br>
    ⏱️ Time: {row['playtime_mean']} min <br>
    🧠 Complexity: {row['averageweight']:.2f}

    </div>

    </div>
    """, unsafe_allow_html=True)

# -------------------------
# RECOMMENDER FUNCTION
# -------------------------
# normalization
scaler_norm = MinMaxScaler()

def recommend_games(players, max_time, complexity, age, favorites):

    current_year = 2026

    if age == "New":
        year_filter = df_user["yearpublished"] >= current_year - 5

    elif age == "Classic":
        year_filter = df_user["yearpublished"] <= current_year - 15

    else:
        year_filter = True


    filtered = df_user[
        (df_user["minplayers"] <= players) &
        (df_user["maxplayers"] >= players) &
        (df_user["playtime_mean"] <= max_time) &
        (df_user["averageweight"] <= complexity) &
        year_filter
    ].copy()


    # Remove expansions and favourite games
    mask = pd.Series(False, index=filtered.index)

    for game in favorites:

        mask = mask | filtered["name"].str.contains(game,case=False,na=False)

    filtered = filtered[~mask]


    # features model
    X = filtered.drop(columns=["id","name","average_rating","rank"])

    # prediction
    filtered["predicted_rating"] = model.predict(X)

    # similarity
    similarity_scores = similarity_from_favorites(favorites)

    filtered["similarity_score"] = similarity_scores[filtered.index]


    filtered["rating_norm"] = scaler_norm.fit_transform(filtered[["predicted_rating"]])
    filtered["similarity_norm"] = scaler_norm.fit_transform(filtered[["similarity_score"]])
    # hybrid score 
    filtered["hybrid_score"] = (0.6 * filtered["rating_norm"] + 0.4 * filtered["similarity_norm"])


    return filtered.sort_values("hybrid_score",ascending=False).head(6)

# -------------------------
# Apply user ratings
# -------------------------

df_user = df.copy()

for _, row in st.session_state.user_ratings.iterrows():

    df_user.loc[
        df_user["name"] == row["name"],
        "average_rating"
    ] = row["user_rating"]


# -------------------------
# Similarity matrix
# -------------------------

similarity_features = df_user.drop(columns=["id","name","average_rating","rank"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(similarity_features)
similarity_matrix = cosine_similarity(X_scaled)


# -------------------------
# Favourites similarity
# -------------------------

def similarity_from_favorites(favorites):

    indices = df_user[df_user["name"].isin(favorites)].index
    scores = similarity_matrix[indices]
    return scores.mean(axis=0)

def similar_games(game_name):

    idx = df_user[df_user["name"] == game_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top = scores[1:6]
    return df_user.iloc[[i[0] for i in top]]

# -------------------------
# TAB 1 RECOMMENDER
# -------------------------

def show_recommender():

    st.title("🎲 What 2 Play?")
    st.header("📖 Boardgame Recommender")
    st.subheader("Find your next favorite boardgame")

    with st.form("recommendation_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            players = st.slider("Number of players",1,10,3)

        with col2:
            max_time = st.slider("Max Playing Time (minutes)",30,300,120)

        with col3:
            complexity = st.slider("Complexity",1.0,5.0,3.0)

        age = st.selectbox("Age of the game",["Any","New","Classic"])

        favorites = st.multiselect("Select some games you like",df_user["name"].values)

        submitted = st.form_submit_button("Recommend", type="primary")

        if submitted:

            if len(favorites) == 0:

                st.warning("⚠️ Please select at least one favorite game.")

            else:
                progress = st.progress(0)

                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                recs = recommend_games(
                    players,
                    max_time,
                    complexity,
                    age,
                    favorites
                )

                st.success("Recommendations ready!")

                cols = st.columns(3)

                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i % 3]:
                        show_game_card(row)

# -------------------------
# TAB RATE
# -------------------------

def show_rating():

    st.title("🎲 What 2 Play?")
    st.header("⭐ Rate a Board Game")
    st.subheader("Help personalize your recommendations")

    game = st.selectbox("Selet a game", df_user["name"].values)

    rating = st.slider("Your rating",1.0,10.0,8.0)

    if st.button("Save rating", type="primary"):

        # remove previous rating
        st.session_state.user_ratings = st.session_state.user_ratings[
            st.session_state.user_ratings["name"] != game
        ]

        new_rating = pd.DataFrame({
            "name":[game],
            "user_rating":[rating]
        })

        st.session_state.user_ratings = pd.concat(
            [st.session_state.user_ratings,new_rating],
            ignore_index=True
        )

        st.session_state.user_ratings.to_csv(
            f"../data/users/{user}_ratings.csv",
            index=False
        )

        st.success(f"Rating for {game} saved ⭐")

# -------------------------
# PAGE ROUTER
# -------------------------

if page == "📖 Recommend a Game":
    show_recommender()

elif page == "⭐ Rate a Game":
    show_rating()