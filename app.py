import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Page settings
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛍️"
)

# Title
st.title("🛍️ AI Shopping Assistant")
st.write("Find the right products using AI-powered search.")


# Product data
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
        55000,
        75000,
        25000,
        3000,
        2500,
        1200,
        1800
    ]
}


# Create DataFrame
df = pd.DataFrame(products)


# Display products
st.subheader("🛒 Available Products")
st.dataframe(df, use_container_width=True)


# Load AI model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# User search
query = st.text_input(
    "🔍 What product are you looking for?"
)


# Search button
if st.button("Find Products"):

    if query.strip():

        # Create product embeddings
        product_embeddings = model.encode(
            df["Description"].tolist(),
            convert_to_tensor=True
        )

        # Create query embedding
        query_embedding = model.encode(
            query,
            convert_to_tensor=True
        )

        # Calculate similarity
        scores = util.cos_sim(
            query_embedding,
            product_embeddings
        )[0]

        # Add similarity scores
        df["Similarity"] = scores.cpu().numpy()

        # Get top 5 products
        results = df.sort_values(
            "Similarity",
            ascending=False
        ).head(5)

        # Display results
        st.subheader("✨ Recommended Products")
        st.dataframe(
            results,
            use_container_width=True
        )

    else:
        st.warning(
            "Please enter a product requirement."
        )