import pandas as pd
import numpy as np
import streamlit as st


@st.cache()
def search_df(df: pd.DataFrame, substring: str, case: bool = False) -> pd.DataFrame:
    mask = np.column_stack(
        [
            df[col].astype(str).str.contains(substring.lower(), case=case, na=False)
            for col in df
        ]
    )
    return df.loc[mask.any(axis=1)]


def plot_embeddings(df: pd.DataFrame) -> None:
    """see: https://plotly.com/python/t-sne-and-umap-projections/"""

    # -- header
    txt = """ 
    <div style="text-align: center; border-style: outset;">
    <a style="text-decoration:none;" href=https://plotly.com/python/t-sne-and-umap-projections>INTERACTIVE 
    VISUALIZATION OF RESULTS</a>
    </div>
    <br>
    """
    st.markdown(txt, unsafe_allow_html=True)

    pd.options.plotting.backend = "plotly"

    from sklearn.manifold import TSNE
    from umap import UMAP
    import plotly.express as px

    def plot_projections(proj_2d, proj_3d):
        fig_2d = px.scatter(proj_2d, x=0, y=1, color=df.label)
        fig_3d = px.scatter_3d(proj_3d, x=0, y=1, z=2, color=df.label)
        st1, st2 = st.columns(2)
        with st1:
            # fig_3d.update_layout(showlegend=False)
            st.plotly_chart(fig_3d, use_container_width=True)
        with st2:
            st.plotly_chart(fig_2d, use_container_width=True)

    features = df.loc[:, df.columns != "label"]

    # -- UMAP Plots -- #
    with st.spinner("fitting umap model ..."):
        umap_2d = UMAP(n_components=2, init="random", random_state=0)
        umap_3d = UMAP(n_components=3, init="random", random_state=0)
        proj_2d = umap_2d.fit_transform(features)
        proj_3d = umap_3d.fit_transform(features)

    txt = """
    > ### [Uniform Manifold Approximation and Projection](https://johnhw.github.io/umap_primes/index.md.html)
    > i.e. Topological Approach

    For more details, see [this video](https://www.youtube.com/embed/nq6iPZVUxZU). 
    """
    st.button("UMAP", help=txt, disabled=True)
    plot_projections(proj_2d, proj_3d)

    st.markdown("<hr>", unsafe_allow_html=True)

    # -- TSNE Plots -- #
    with st.spinner("fitting tsne model ..."):
        perplexity = min(30, features.shape[0] - 1)
        tsne_2d = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        tsne_3d = TSNE(n_components=3, random_state=0, perplexity=perplexity)
        import sklearn
        st.write(sklearn.__version__)
        proj_2d = tsne_2d.fit_transform(features)
        proj_3d = tsne_3d.fit_transform(features)

    txt = """
    > ### [t-distributed Stochastic Neighbor Embedding](https://scikit-learn.org/stable/modules/manifold.html#t-sne)
    > i.e. Local Structure

    For more details, see [this video](https://www.youtube.com/embed/RJVL80Gg3lA).
    """
    st.button("t-SNE", help=txt, disabled=True)
    plot_projections(proj_2d, proj_3d)
