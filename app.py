import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛍️"
)

st.title("🛍️ AI Shopping Assistant")
st.write("Find the right products using AI-powered search.")

products = {
    
    "Product": [
        "Laptop",
        "Gaming Laptop",
        "Smartphone",
        "Headphones",
        "Keyboard",
        "Mouse",
        "Laptop Bag"
    ],
    "Description": [
        "Powerful laptop for programming coding and students",
        "High performance laptop for gaming programming and development",
        "Smartphone with good camera and battery life",
        "Wireless headphones for music and online classes",
        "Mechanical keyboard for programming and gaming",
        "Wireless mouse for computer and laptop",
        "Laptop backpack for college students"
    ],
    "Price": [
        55000, 75000, 25000, 3000, 2500, 1200, 1800
    ]
}

df = pd.DataFrame(products)

st.subheader("🛒 Available Products")
st.dataframe(df, use_container_width=True)

query = st.text_input("🔍 What product are you looking for?")

if st.button("Find Products"):

    if query.strip():

        model = SentenceTransformer("all-MiniLM-L6-v2")

        product_embeddings = model.encode(
            df["Description"].tolist(),
            convert_to_tensor=True
        )

        query_embedding = model.encode(
            query,
            convert_to_tensor=True
        )

        scores = util.cos_sim(
            query_embedding,
            product_embeddings
        )[0]

        df["Similarity"] = scores.cpu().numpy()

        results = df.sort_values(
            "Similarity",
            ascending=False
        ).head(5)

        st.subheader("✨ Recommended Products")
        st.dataframe(results, use_container_width=True)

    else:
        st.warning("Please enter a product requirement.")