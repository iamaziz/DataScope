from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd  # TODO: import modin.pandas as pd
import streamlit as st
from annoy import AnnoyIndex
from pandas_profiling import ProfileReport
from pandasql import sqldf
from pydataset import data
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from st_aggrid import AgGrid
from streamlit.components.v1 import iframe
from streamlit_pandas_profiling import st_profile_report
from stvis import pv_static

from src.utils import plot_embeddings, search_df

st.set_page_config(page_title="DataScope", layout="wide", page_icon="🔬")


def header():
    *_, st5 = st.columns(13)
    with st.container():
        """
        # _DataScope🔬_ TABULAR DATA ANALYZER 🔍
        """

    with st.expander("ABOUT"):
        msg = """
        
        <div style="text-align: center; border-style: outset">
        
        TL;DR **Seeing tabular sample_data from a different angle**
         
        <a style="font-size: 5px" href="https://media.giphy.com/media/ftAyb0CG1FNAIZt4SO/giphy.gif"><img src="https://media.giphy.com/media/ftAyb0CG1FNAIZt4SO/giphy.gif" /></a>
        
        </div>
     
        <div style="font-family: arial">
        
        **What?**
        
        This tool analyzes a tabular dataset to
        > 1) Understand relationships between variables "columns" (**Directed Graphs**)
        > 2) Discover Patterns / Semantic Search  (**Semantic Similarity Space**)

        Also, it provides
        > 3. Dataset Profiler
        > 4. Customized _Feature Embeddings_ (the output of number 2 above)
        
        `TODO: more docs .. to come`
        
        > <sub> `<️WIP>⚠️ ... can be buggy on some edge cases; however, it works.`</sub>
        
        <hr>
        
        **Getting Started**
        
        To start, upload a new dataset or select a sample dataset from the list.
                
        </div>
        """
    h_msg = """
    To load a new dataset, refresh the browser.
    """
    st5.button(label="Help", disabled=True, help=h_msg)
    with st.expander("README"):
        st.markdown(msg, unsafe_allow_html=True)

    style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def _read_csv(f, **kwargs):
    df = pd.read_csv(f, on_bad_lines="skip", **kwargs)
    # clean
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache(allow_output_mutation=True)
def _read_excel(f, **kwargs):
    return pd.read_excel(f, **kwargs)


def read_pydataset():
    available_datasets = data()
    st.write(available_datasets)
    options = [""] + available_datasets["dataset_id"].tolist()
    choice = st.selectbox(label="Select dataset to use", options=options)
    if choice:
        return data(choice)
    st.stop()


def load_dataset(container):
    options = [
        "",
        "Arxiv ML Papers",
        "Periodic Table",
        "Companies",
        "Iris Flowers",
        "Movies",
        "Music Artists",
        "Soccer Players",
        "PyDataset (Explore more datasets)",
    ]

    with container:
        st1, st2 = st.columns(2)

    with st1:
        choice = st.selectbox(
            "Choose a sample dataset",
            options=options,
        )
        if choice.startswith("PyDataset"):
            return read_pydataset()

    # -- Upload file dataset
    with st2:
        txt = "Upload a dataset (supported files: .csv, .tsv, .xls, .xlsx)"
        uploaded = st.file_uploader(label=txt)
        choice = uploaded.name if uploaded else choice

    with st2:
        url = st.text_input("Or read from a url file (supported: .csv and .tsv)", placeholder="Enter URL")
        if url:
            return _read_csv(url)

    st.session_state.fit_columns = False if choice.endswith(".csv") else True
    st.session_state.file_name = choice
    params = st2.text_input("", placeholder="sep && skip_rows").split("&&")
    if uploaded:
        sep = params[0].strip()
        sep = sep if sep else ","
        skip = 0 if len(params) < 2 else int(params[1].strip())
        if uploaded.name.endswith((".csv", ".tsv")):
            return _read_csv(uploaded, sep=sep, skiprows=skip)
        if uploaded.name.endswith(".xlsx"):
            return _read_excel(uploaded, skiprows=skip)

    # -- read file
    if choice.startswith("Arxiv"):
        return _read_csv(f"./sample_data/arxiv_papers/arxivData_flat.csv")
    if choice.startswith("Companies"):
        return _read_csv("./sample_data/companies-dataset/companies_sorted_SMALL.csv")
    if choice.startswith("Movies"):
        url = "https://raw.githubusercontent.com/reisanar/datasets/master/HollywoodMovies.csv"
        return _read_csv(url)
    if choice.startswith("Periodic"):
        url = "https://gist.githubusercontent.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee/raw/1d92663004489a5b6926e944c1b3d9ec5c40900e/Periodic%2520Table%2520of%2520Elements.csv"
        return _read_csv(url)
    if choice.startswith("Iris"):
        url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
        return _read_csv(url)
    if choice.startswith("Music"):
        path = "sample_data/music-artists-popularity/artist_truncated.csv"
        return _read_csv(path)
    if choice.startswith("Soccer"):
        choice = "/players/players_20.csv"

    # st.session_state.trained_model = False
    try:
        return _read_csv(f"./sample_data/{choice}")
    except ValueError:
        return _read_excel(f"./sample_data/{choice}")
    except:
        st.stop()


def render_as_aggrid(_df: pd.DataFrame) -> pd.DataFrame:
    returned = AgGrid(
        data=_df,
        data_return_mode="FILTERED",
        fit_columns_on_grid_load=False,  # st.session_state.fit_columns,
        enable_enterprise_modules=True,
        theme="dark",
    )
    _df = returned.data
    return _df


def _st_search_dataframe(df, key):
    if term := st.text_input(
            "",
            placeholder="Type to search all records/any column (case-insensitive)",
            help="Global Search i.e. find all rows that contain the searched keyword in any column.",
            value="",
            key=key,
    ):
        return _search_dataframe(df, term)
    return df


@st.cache()
def _search_dataframe(df, term: str) -> pd.DataFrame:
    return search_df(df, substring=term)


def st_query_dataframe(df, key):
    if query := st.text_input(
            "",
            placeholder="""select * form df limit 10;""",
            help="The SQL query to execute against the dataframe below (by default: 'df' is the table name)",
            value="",
            key=key,
    ):
        with st.spinner("running sql_query.."):
            return _sql_query_df(df, query)
    return df


@st.cache(suppress_st_warning=True)
def _sql_query_df(df, query_str: str) -> pd.DataFrame:
    try:
        return sqldf(query_str, locals())
    except:  # pd.PandaSQLException:
        st.warning("Invalid SQL query!", icon="⚠️")
        return df


class Helper:
    @staticmethod
    def tensorflow_projector_frame():
        """Embed the TensorBoard Projector to allow uploading and visualizing local results"""
        st.caption("Use 'Load' button to upload the result .tsv files to TensorBoard")
        iframe("https://projector.tensorflow.org/", height=900, scrolling=True)

    @staticmethod
    def st_display_dataframe(
            _df: pd.DataFrame,
            save_as="results.csv",
            allow_search=True,
            key=None,
    ) -> pd.DataFrame:

        # -- layout display
        st1, st2 = st.columns(2)
        help_ = """
         'aggrid' is excel-like view with column-wise filtering. 
         The other two are static and basic layout.
         """
        view = st2.radio(
            options=["aggrid", "dataframe", "table"],
            label="view as",
            horizontal=True,
            help=help_,
            key=f"view_{key}",
        )
        # -- search and/or filtering
        st1, st2 = st.columns(2)
        if allow_search:
            with st1:
                _df = _st_search_dataframe(_df, key=f"find_{key}")
            with st2:
                _df = st_query_dataframe(_df, key=f"query_{key}")

        if view == "aggrid":
            _df = render_as_aggrid(_df=_df)
        else:
            getattr(st, view)(_df.astype(str))
        st.markdown(f"> `{_df.shape[0]} records`", unsafe_allow_html=True)

        # -- to download
        @st.cache
        def convert_df(_df):
            return _df.to_csv().encode("utf-8")

        csv = convert_df(_df)
        st.download_button(
            "Download",
            csv,
            save_as,
            "text/csv",
        )
        return _df


class UI:
    def __init__(self, data: pd.DataFrame) -> None:

        # -- main sample_data display
        df = Helper.st_display_dataframe(data)

        # -- Expanders
        with st.expander("NETWORK GRAPHIZER"):
            NetworkGraphizer(df)
        with st.expander("SEMANTIC SPACE AND SEARCH"):
            EmbeddingApp(df)
            # st.markdown(st.session_state)
            if "trained_model" in st.session_state and st.session_state.trained_model:
                SimilaritySearch(df_=df, st_key="local")

            st.markdown("<hr>", unsafe_allow_html=True)
            st1, st2 = st.columns(2)
            if st1.checkbox("3D VISUALIZE THE SIMILARITY SPACE MODEL"):
                Helper.tensorflow_projector_frame()
            if st2.checkbox("UPLOAD PREVIOUS RESULTS"):
                SimilaritySearch(df_=df, external_files=True, st_key="upload")
        with st.expander("DATA PROFILING REPORT"):
            DataProfiler(df)


class NetworkGraphizer:
    """Build PyVis interactive network based on the input dataframe"""

    def __init__(self, df: pd.DataFrame, max_rows: int = 1000):
        if df.shape[0] < 1 or df.shape[1] < 2:
            return

        if df.shape[0] > max_rows:
            m = f"Dataframe {df.shape} is too large, truncating to top {max_rows} records"
            st.warning(m)
            df = df[:max_rows]

        label, run, source, target = self._user_input(df)
        if run:
            network = self.graphizer_data(df, label, source, target)
            # render the network in st
            pv_static(network.g)
            # -- download graph network
            self._download_graph_network("graph.html")

    @staticmethod
    def _user_input(df):
        st.write(df.shape)
        cols = df.columns
        cols_rev = df.columns[::-1]
        # select source/target nodes from the df columns and edge labels
        st0, st1, st2, st3 = st.columns(4)
        run = st0.button("render")
        source = st1.multiselect("source", options=cols, default=cols[0])
        target = st2.multiselect("target", options=cols_rev, default=cols[-2])
        label = st3.multiselect("label", options=cols_rev, default=cols[-1])
        return label, run, source, target

    @staticmethod
    def graphizer_data(df, label, source, target):
        n_labels = len(label)
        NetworkGraphizer._assert_valid_inputs(label, source, target)

        # collect graph components i.e. nodes and labels - based on user selections
        # [(col_name, col_name:: values), ... etc]
        heads = [
            # (c, f"{c}:: {vals}")
            (c, f"{vals}")
            for c in source
            for vals in df[c].astype(str).tolist()
        ]
        # [(col_name, col_name:: values), ... etc]
        tails = [
            # (c, f"{c}:: {vals}")
            (c, f"{vals}")
            for c in target
            for vals in df[c].astype(str).tolist()
        ]
        # Labels values
        labels = [f"{txt}" for c in label for txt in df[c].tolist()]
        # -- segmenting labels eg
        # from: [col1_val1, col1_val2, ... col1_valN, col2_val1, col2_val2, ...col2_valN, ... etc]
        # to:   ["col1_val1 col2_val1 ..colN_val1", "col1_val2 col2_val2 ... colN_val1", ... etc]

        # -- aligning labels with edge-type
        idx_parts = list(range(0, len(labels) + 1, len(labels) // n_labels))
        chunks = [
            labels[idx_parts[i]: idx_parts[i + 1]] for i in range(len(idx_parts) - 1)
        ]
        aligned_parts_labels = list(zip(*[c for c in chunks]))
        concat_labels = [" | ".join(foo) for foo in aligned_parts_labels]
        labels = concat_labels
        # --

        # add column name to the graph edges
        # labels = [f"{label}:: {l}" for l in labels]

        # -- build the directed graph network
        network = NetworkVisualizer(df.shape[0], heads, tails, labels)
        return network

    @staticmethod
    def _assert_valid_inputs(label, source, target):
        # validate inputs
        def warn(msg):
            st.warning(f"Please choose `{msg}` column(s)")
            st.stop()

        assert len(source) > 0, warn("source")
        assert len(target) > 0, warn("target")
        assert len(label) > 0, warn("label")

    @staticmethod
    def _download_graph_network(output_file_name):
        with open("./graph.html") as f:
            m = "Download the generated graph"
            st.download_button(m, f, file_name=output_file_name)


class NetworkVisualizer:
    def __init__(self, num_records, heads_, tails, labels):
        # https://pyvis.readthedocs.io/

        # flatten
        tails_ = [t[1] for t in tails]
        tails_labels = [t[0] for t in tails]
        heads = [h[1] for h in heads_]

        # TODO: 5 is the MAX NUM of targets (hard-coded)
        edges = [(h, t, r) for h, t, r in zip(heads * 5, tails_, tails_labels)]
        labeled_edges = dict(zip(edges, labels * 5))

        # -- tail_node properties eg coloring, shape, groups
        _tails_names = set(t[0] for t in tails)
        SUPPORTED_COLORS = ["orange", "green", "purple", "blue", "black"]
        colors = {n: c for n, c in zip(_tails_names, SUPPORTED_COLORS)}
        SUPPORTED_SHAPES = ["triangle", "diamond", "star", "square", "hexagon"]
        shapes = {n: c for n, c in zip(_tails_names, SUPPORTED_SHAPES)}
        groups = dict(zip(_tails_names, range(len(_tails_names))))
        # legend info
        info_color_shape_map = shapes.copy()
        for k, v in colors.items():
            info_color_shape_map[
                k
            ] = f"{colors[k].capitalize()}{shapes[k].capitalize()}"

        # see: https://visjs.github.io/vis-network/docs/network/nodes.html

        n_nodes = len(set(heads) | set(tails_))  # == g.num_nodes()
        n_edges = len(edges)  # == g.num_edges()
        stats = f"{num_records} records \n({n_nodes} nodes, {n_edges} edges)"

        g = Network(
            height="1200px",
            width="1800px",
            heading=f"Networked Results of: {stats}",
            directed=True,
            layout=False,  # True forces hierarchical layers
            bgcolor="#c9dcde",
        )
        # -- Network properties
        # Physics Solver option - for stabilized graph layout
        # g.barnes_hut()
        g.force_atlas_2based(central_gravity=0.005, spring_length=445, overlap=0.35)

        # -- add graph contents

        # create nodes
        for node in set(heads):
            if node == "nan":
                continue
            g.add_node(node, shadow=True)

        for k, node in tails:
            if node == "nan":
                continue
            degree = tails_.count(node)
            size = min(degree, 55)  # upper bound for node size
            label = f"{node} (deg. {degree})"
            g.add_node(
                node,
                label=label,
                color=colors[k],
                shape=shapes[k],
                size=max(size, 15),
                # group=groups[k],
                borderWidth=30,
                font="45px Courier red",
                hover=True,
            )

        # create edges between nodes
        for edge, edge_label in labeled_edges.items():
            if "nan" in edge:
                continue
            h, t, r = edge
            g.add_edge(source=h, to=t, title=f"{edge_label} ('{r}')", shadow=True)

        # g.show_buttons(filter_=["physics", "nodes", "edges"])
        g.show_buttons(
            filter_=[
                "nodes",
                "edges",
                "layout",
                "interaction",
                "manipulation",
                "physics",
                "selection",
                "renderer",
            ]
        )
        # g.show_buttons(filter_=['layout', 'interaction', 'manipulation', 'selection', 'renderer'])

        # Graph's info and "Legend node"
        graph_title = f"Graph Network\n({g.num_nodes()} nodes, {g.num_edges()} edges)\nBased on\n{num_records} rows of '{st.session_state.file_name}'"
        g.add_node(graph_title, shape="text", font="80px Optima blue", color="#d9eff2")
        _unpack = "\n".join(
            [f"{v}: '{k}' column" for k, v in info_color_shape_map.items()]
        )
        main_nodes_legend = "Circles: the 'source' column"
        lengend_info = f"LEGEND\n{_unpack}\n{main_nodes_legend}"
        g.add_node(
            lengend_info, shape="box", font="80px Monaco black", color="#2ea8b8"
        )  # , borderWidth=40)

        self.g = g


class EmbeddingApp:
    def __init__(self, df: pd.DataFrame):

        with st.spinner("building dataset.."):
            labels, sentences = self._prepare_dataset(df)
        st1, st2 = st.columns(2)
        # -- pre-trained model choice
        selected_model = st2.selectbox(
            "Choose a pre-trained model",
            ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
            help="The pre-trained embedding model to use for computing similarity vectors. More info, "
                 "see: https://huggingface.co/models?pipeline_tag=sentence-similarity",
        )

        # -- compute embedding for the input
        if not st1.button("Build embedding"):  # or len(sentences) < 3:
            return

        assert labels is not None, (st.warning("Choose label column(s)"), st.stop())

        embedding = self._build_embeddings(labels, selected_model, sentences)

        # -- Save results
        Path("./output").mkdir(exist_ok=True)

        np.savetxt("./output/vecs.tsv", embedding, delimiter="\t")
        with open("./output/metadata.tsv", "w") as f:
            for l in labels:
                f.write(f"{l}\n")

        # -- Download results
        st1, st2 = st.columns(2)
        with open("./output/vecs.tsv") as f:
            st2.download_button(
                "Download embedding/vectors file", f, file_name="embeddings.tsv"
            )
        with open("./output/metadata.tsv") as f:
            st1.download_button(
                "Download tsv metadata (labels) file", f, file_name="labels.tsv"
            )

        # -- set st session variable
        st.session_state.trained_model = True

    def _build_embeddings(self, labels, selected_model, sentences):
        # -- Build Embedding
        # st.caption("INPUT:")
        # st1, st2 = st.columns(2)
        # st1.caption("> Labels")
        # st1.dataframe(labels)
        # st2.caption("> Features (the selected columns)")
        # st2.dataframe(sentences)
        if len(sentences) > 1:
            sentences = list(sentences)
            embedding = self.transform(sentences, selected_model)
        st.caption("> OUTPUT: feature vectors")
        st.write(embedding)
        return embedding

    def _prepare_dataset(self, df):
        st1, st2 = st.columns(2)
        label_cols = st1.multiselect(
            "Choose label column",
            options=[""] + df.columns,
            help="For search and legend labels",
        )
        label_cols = [c.strip() for c in label_cols]
        feature_cols = st2.multiselect(
            "Choose feature columns",
            options=df.columns,
            help="Columns to concatenate as sentences to compute embedding",
        )
        feature_cols = [c.strip() for c in feature_cols]
        if label_cols:
            st1.write(df[label_cols])
        if feature_cols:
            st2.write(df[feature_cols])
        if len(feature_cols) > 0:
            feature_cols = feature_cols if len(feature_cols) > 0 else []
            join_feat_cols = lambda row: " ".join(
                [str(v) for v in row if v is not None]
            )
            join_label_cols = lambda row: " | ".join(
                [str(v).strip() for v in row if v is not None]
            )
            sentences = df[feature_cols].agg(join_feat_cols, axis=1)
            labels = df[label_cols].agg(join_label_cols, axis=1)

            self.label_cols = label_cols
            self.feature_cols = feature_cols
            return labels, sentences
        return None, None

    @st.cache()
    def transform(self, sentences: List[str], pretrained_model: str) -> np.ndarray:
        """compute embeddings for the input sentences.
        see: https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        and: https://huggingface.co/models?pipeline_tag=sentence-similarity
        """
        # sentences = ["This is an example sentence", "Each sentence is converted"]
        with st.spinner(f"Building embedding using '{pretrained_model}' ..."):
            model = SentenceTransformer(f"sentence-transformers/{pretrained_model}")
            embeddings = model.encode(sentences)
        return embeddings


class DataProfiler:
    def __init__(self, df) -> None:
        st1, st2 = st.columns(2)

        msg = "'minimal' is faster to generate, 'detailed' includes `Interactions`, `Correlations`, `Missing Values` .. etc."
        level = st1.radio(
            options=["minimal", "detailed"],
            label="Report level",
            horizontal=True,
            help=msg,
        )
        minimal = True if level == "minimal" else False

        m = "Using [pandas-profiling](https://pandas-profiling.ydata.ai/docs/master/pages/getting_started/concepts.html)"
        if st2.button("Generate Report", help=m):
            profile = ProfileReport(df, title="DataFrame Profiler", minimal=minimal)
            st_profile_report(profile)

            # download the report
            output_file = "dataset_profile_report.html"
            profile.to_file(output_file=output_file)
            with open(output_file) as f:
                m = "Download the generated report"
                st1.download_button(m, f, file_name=output_file)


def _upload_external_files() -> Tuple[np.ndarray, pd.DataFrame]:
    placeholder = st.empty()
    with placeholder:
        st.caption("Upload files to query similarity (i.e. Semantic Search)")
        st1, st2 = st.columns(2)
        vectors_file = st1.file_uploader(label="upload vectors")
        labels_file = st2.file_uploader(label="upload labels")

        if vectors_file and labels_file:
            embeddings = np.loadtxt(vectors_file, dtype="float32")
            labels = _read_csv(labels_file, sep="\t", names=["element"])
            placeholder.empty()
            return embeddings, labels

    return None, None


class SimilaritySearch:
    def __init__(
            self, df_: pd.DataFrame, external_files: bool = False, st_key: str = "any"
    ):

        self.st_key = st_key

        if external_files:
            embeddings, labels = _upload_external_files()
        else:
            embeddings, labels = self._load_local_results()

        if isinstance(embeddings, np.ndarray) and isinstance(labels, pd.DataFrame):
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("> ### Similarity Search")
            t = self._init_tree(embeddings)
            st.markdown(f"> `Search space: {len(embeddings)} records (labels/vectors)`")

            # -- search term
            term, top_n = self.what_to_search(labels)
            if term:
                st.write("> ### results")
                self._execute_query(term, df_, embeddings, t, labels, top_n)

    def _load_local_results(self) -> Tuple[np.ndarray, pd.DataFrame]:
        embeddings = self._load_embedding()
        labels = self._load_labels()
        return embeddings, labels

    @staticmethod
    @st.experimental_singleton
    def _init_tree(embeddings):
        t = SimilaritySearch.build_annoy_index(embeddings)
        return t

    def _execute_query(self, term, df, embeddings, t, labels, top_n):

        idx = labels.loc[labels["element"] == term].index

        # -- search
        res_idx, distances = t.get_nns_by_item(idx[0], top_n, include_distances=True)

        # -- show results
        res_labels = labels.iloc[res_idx]
        try:
            results = df.iloc[res_idx]
            results.insert(0, "<<SIMILARITY>>", distances)
            Helper.st_display_dataframe(
                results,
                save_as=f"similarity_results_for-{term}.csv",
                key=f"results_{self.st_key}",
            )

            # -- plot similarity results as 3D-plotly with TSNE
            all_vecs = pd.DataFrame(embeddings)
            res_vecs = all_vecs.iloc[res_idx]
            res_vecs["label"] = res_labels
            with st.spinner("Generating result plots .."):
                plot_embeddings(res_vecs)

        except IndexError:
            st.warning(
                "Please ensure the loaded files (embeddings and labels) are associated with the current dataframe."
            )
            st.stop()

    @staticmethod
    def _load_embedding(f="./output/vecs.tsv") -> np.ndarray:
        """load the computed embeddings"""
        return np.loadtxt(f, dtype="float32")

    @staticmethod
    def _load_labels(f="./output/metadata.tsv") -> pd.DataFrame:
        return _read_csv(f, sep="\t", names=["element"])

    @staticmethod
    def build_annoy_index(data: np.ndarray):
        """https://github.com/spotify/annoy#python-code-example"""

        i, j = data.shape

        t = AnnoyIndex(j, "euclidean")
        for i, v in enumerate(data):
            t.add_item(i, v)
        t.build(10)
        return t

    def what_to_search(self, df: pd.DataFrame):
        st1, st2 = st.columns(2)
        term = st1.selectbox(
            "Find similar to",
            options=[None] + list(df["element"].values),
            key=f"term_{self.st_key}",
        )
        top_n = st2.slider(
            "Number of results",
            min_value=5,
            value=10,
            max_value=50,
            help="Slide for to set number of results to return (default: 20).",
            key=f"slider_{self.st_key}",
        )
        if term:
            return term, top_n
        return None, None


def main():
    header()
    placeholder = st.empty()
    _df = load_dataset(placeholder)
    placeholder.empty()
    UI(_df)


if __name__ == "__main__":
    main()
