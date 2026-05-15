import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


STANDARD_COLUMNS = [
    "label",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass", "jet1_btag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass", "jet2_btag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass", "jet3_btag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_mass", "jet4_btag",
]


def default_filenames(n_files_to_use, directory="evenet-test/evenet-ma30"):
    return [
        f"{directory}/data_Combined_Balanced_run_{run}.parquet"
        for run in range(n_files_to_use)
    ]


def build_column_map(n_jets=4):
    column_map = {"classification": "label"}
    for jet_index in range(n_jets):
        source_index = jet_index
        jet = jet_index + 1
        column_map[f"x:{source_index}:1"] = f"jet{jet}_pt"
        column_map[f"x:{source_index}:2"] = f"jet{jet}_eta"
        column_map[f"x:{source_index}:3"] = f"jet{jet}_phi"
        column_map[f"x:{source_index}:4"] = f"jet{jet}_btag"
        column_map[f"derived_from_x:{source_index}:0"] = f"jet{jet}_mass"
    return column_map


def has_evenet_vector_columns(df, label_column, n_jets=4):
    required = [label_column]
    for i in range(n_jets):
        required += [f"x:{i}:0", f"x:{i}:1", f"x:{i}:2", f"x:{i}:3", f"x:{i}:4", f"x_mask:{i}"]
    return all(col in df.columns for col in required)


def make_standard_event_dataframe(df, label_column, n_jets=4):
    if set(STANDARD_COLUMNS).issubset(df.columns):
        return df[STANDARD_COLUMNS].copy()

    if not has_evenet_vector_columns(df, label_column=label_column, n_jets=n_jets):
        missing = []
        for i in range(n_jets):
            for col in [f"x:{i}:0", f"x:{i}:1", f"x:{i}:2", f"x:{i}:3", f"x:{i}:4", f"x_mask:{i}"]:
                if col not in df.columns:
                    missing.append(col)
        raise AssertionError(
            "Required flat jet columns were not found. "
            "Inspect the dataset schema and flatten the four leading jets. "
            f"Missing examples: {missing[:10]}"
        )

    out = pd.DataFrame(index=df.index)
    out["label"] = df[label_column].astype(int)

    for i in range(n_jets):
        jet = i + 1
        valid = df[f"x_mask:{i}"].astype(bool)
        log_energy = df[f"x:{i}:0"].astype(float)
        log_pt = df[f"x:{i}:1"].astype(float)
        eta = df[f"x:{i}:2"].astype(float)
        phi = df[f"x:{i}:3"].astype(float)
        btag = df[f"x:{i}:4"].astype(float)

        pt = np.exp(log_pt)
        energy = np.exp(log_energy)
        pz = pt * np.sinh(eta)
        mass2 = energy**2 - pt**2 - pz**2
        mass = np.sqrt(np.clip(mass2, 0, None))

        out[f"jet{jet}_pt"] = pt.where(valid, np.nan)
        out[f"jet{jet}_eta"] = eta.where(valid, np.nan)
        out[f"jet{jet}_phi"] = phi.where(valid, np.nan)
        out[f"jet{jet}_mass"] = mass.where(valid, np.nan)
        out[f"jet{jet}_btag"] = btag.where(valid, np.nan)

    return out


def build_cutflow(events):
    jet_pt_columns = [f"jet{i}_pt" for i in range(1, 5)]
    jet_btag_columns = [f"jet{i}_btag" for i in range(1, 5)]

    four_jet_mask = events[jet_pt_columns].notna().all(axis=1)
    pt_mask = four_jet_mask & (events[jet_pt_columns] > 30).all(axis=1)
    n_btags_raw = events[jet_btag_columns].fillna(0).round().astype(int).sum(axis=1)
    btag_mask = pt_mask & (n_btags_raw >= 3)

    selection_masks = {
        "initial events": pd.Series(True, index=events.index),
        "after four-jet requirement": four_jet_mask,
        "after pT requirement": pt_mask,
        "after b-tag requirement": btag_mask,
    }

    cutflow = pd.DataFrame({
        "all": [int(mask.sum()) for mask in selection_masks.values()],
        "signal": [int((mask & (events["label"] == 1)).sum()) for mask in selection_masks.values()],
        "background": [int((mask & (events["label"] == 0)).sum()) for mask in selection_masks.values()],
    }, index=selection_masks.keys())
    return cutflow, btag_mask


def jets_are_pt_ordered(df):
    return (
        (df["jet1_pt"] >= df["jet2_pt"]) &
        (df["jet2_pt"] >= df["jet3_pt"]) &
        (df["jet3_pt"] >= df["jet4_pt"])
    )


def sort_four_jets_by_pt(df):
    sorted_rows = []
    jet_fields = ["pt", "eta", "phi", "mass", "btag"]
    for _, row in df.iterrows():
        jets = []
        for jet in range(1, 5):
            jets.append({field: row[f"jet{jet}_{field}"] for field in jet_fields})
        jets = sorted(jets, key=lambda item: item["pt"], reverse=True)

        new_row = row.copy()
        for jet, values in enumerate(jets, start=1):
            for field in jet_fields:
                new_row[f"jet{jet}_{field}"] = values[field]
        sorted_rows.append(new_row)
    return pd.DataFrame(sorted_rows).reset_index(drop=True)


def load_physics_pipeline(
    repo_id="Avencast/EveNet-ExoticHiggs-H2a4b",
    n_files_to_use=24,
    max_events=200_000,
    random_state=123,
    directory="evenet-test/evenet-ma30",
):
    filenames = default_filenames(n_files_to_use=n_files_to_use, directory=directory)

    dataframes = []
    for filename in filenames:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        one_file = pd.read_parquet(path)
        one_file["source_file"] = filename
        dataframes.append(one_file)

    data = pd.concat(dataframes, ignore_index=True)
    if len(data) > max_events:
        data = data.sample(n=max_events, random_state=random_state).reset_index(drop=True)

    label_column = "classification" if "classification" in data.columns else "label"
    assert len(data) > 0, "The dataframe is empty. Check the downloaded files."
    assert label_column in data.columns, (
        f"Could not find a label column. Looked for {label_column!r}. "
        "Inspect data.columns and update label_column."
    )

    label_values = set(data[label_column].dropna().astype(int).unique())
    assert label_values <= {0, 1}, f"Labels must be only 0 and 1, found {sorted(label_values)}."
    assert label_values == {0, 1}, f"Both classes must be present, found {sorted(label_values)}."

    events = make_standard_event_dataframe(data, label_column=label_column)
    missing_standard = sorted(set(STANDARD_COLUMNS) - set(events.columns))
    assert not missing_standard, f"Missing standard columns: {missing_standard}"
    assert events["label"].isin([0, 1]).all(), "Standard label column must contain only 0 and 1."
    assert events["label"].nunique() == 2, "Standard dataframe must contain both classes."

    cutflow, selected_mask = build_cutflow(events)
    selected_events = events.loc[selected_mask].reset_index(drop=True)
    assert len(selected_events) > 0, "No events survived the basic selection."
    assert selected_events["label"].nunique() == 2, "Selection removed one class. Use more files or loosen cuts."

    ordered_mask = jets_are_pt_ordered(selected_events)
    fraction_ordered_before_sort = float(ordered_mask.mean())
    if not ordered_mask.all():
        selected_events = sort_four_jets_by_pt(selected_events)
        assert jets_are_pt_ordered(selected_events).all(), "Jet sorting failed. Check the pT columns."

    column_summary = pd.DataFrame({
        "index": range(len(data.columns)),
        "column": data.columns,
        "dtype": [str(dtype) for dtype in data.dtypes],
    })
    label_counts = data[label_column].value_counts().sort_index().rename_axis("label").reset_index(name="events")
    pipeline_flow = pd.DataFrame([
        {
            "stage": "1. Download Parquet files",
            "what happens": "Load a small subset of individual Hugging Face Parquet files.",
            "notebook object": "data",
            "rows": len(data),
        },
        {
            "stage": "2. Inspect raw schema",
            "what happens": "Identify the label column and EveNet vector columns such as x:0:1 and x_mask:0.",
            "notebook object": "column_summary",
            "rows": len(column_summary),
        },
        {
            "stage": "3. Standardize events",
            "what happens": "Convert the first four valid EveNet vectors into jet1-jet4 pt, eta, phi, mass, and btag columns.",
            "notebook object": "events",
            "rows": len(events),
        },
        {
            "stage": "4. Apply physics cuts",
            "what happens": "Require four jets, pT > 30 GeV for the four jets, and at least three b-tagged jets.",
            "notebook object": "cutflow",
            "rows": len(selected_events),
        },
        {
            "stage": "5. Sort jets by pT",
            "what happens": "Ensure jet1 is the leading jet, jet2 is next, and so on, so ML columns have fixed meaning.",
            "notebook object": "selected_events",
            "rows": len(selected_events),
        },
    ])

    return {
        "repo_id": repo_id,
        "filenames": filenames,
        "label_column": label_column,
        "data": data,
        "events": events,
        "selected_events": selected_events,
        "load_summary": pd.DataFrame({
            "files_loaded": [len(filenames)],
            "events_loaded": [len(data)],
            "max_events": [max_events],
        }),
        "data_shape": pd.DataFrame({"rows": [data.shape[0]], "columns": [data.shape[1]]}),
        "column_summary": column_summary,
        "label_counts": label_counts,
        "pipeline_flow": pipeline_flow,
        "column_map": pd.DataFrame(build_column_map().items(), columns=["dataset_column", "standard_name"]),
        "standard_event_shape": pd.DataFrame({"rows": [events.shape[0]], "columns": [events.shape[1]]}),
        "cutflow": cutflow,
        "jet_ordering": pd.DataFrame({
            "fraction_ordered_before_sort": [fraction_ordered_before_sort],
            "sorted_if_needed": [not bool(ordered_mask.all())],
        }),
    }
