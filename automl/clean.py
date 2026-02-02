import pandas as pd
from ydata_profiling import ProfileReport


def preprocess_from_profile(df, corr_threshold=0.85):
    """
    Nettoyage automatique du dataset avec ydata-profiling (profiling léger)
    """

    # ============================================================
    # SAFE MODE : ne pas lancer ydata_profiling si trop de colonnes
    # ============================================================
    if df.shape[1] > 2000:
        print(f"[WARN] Profiling désactivé (SAFE MODE) — {df.shape[1]} colonnes.")
        # Clean minimal sans profiling
        df = df.copy()

        # On supprime les colonnes qui n'ont qu'une seule valeur (inutiles pour le modèle)
        nunique = df.nunique()
        constant_cols = nunique[nunique == 1].index.tolist()
        df.drop(columns=constant_cols, inplace=True, errors="ignore")

       # Remplissage basique des trous : médiane pour les chiffres, mode pour le texte
        num_cols = df.select_dtypes(include=[float, int]).columns
        cat_cols = df.select_dtypes(include=["object"]).columns

        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        for col in cat_cols:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")

        return df

    # ============================================================
    # STANDARD MODE : Analyse fine avec ydata-profiling
    # ============================================================
    # On lance l'analyseur. 'minimal=True' est crucial ici : cela évite les calculs 
    # trop lourds qui ne servent pas au nettoyage automatique.
    profile = ProfileReport(
        df,
        title="AutoML Clean Profile",
        explorative=True,
        minimal=True
    )

    # On récupère toutes les stats calculées par le rapport
    desc = profile.get_description()
    variables = desc.variables  

    df = df.copy()

    # === GESTION DES VALEURS MANQUANTES ===
    for col, stats in variables.items():
        if col not in df.columns:
            continue

        missing_perc = stats["p_missing"] * 100
        col_type = stats["type"]
        # Si la colonne est vide à plus de 30%, on la jette, c'est irrécupérable.Sinon, on essaye de boucher les trous
        if missing_perc > 30:
            df.drop(columns=[col], inplace=True)
        elif 5 < missing_perc <= 30:
            if col_type == "Numeric":
                df[col] = df[col].fillna(df[col].median())# La médiane est plus sûre que la moyenne (moins sensible aux valeurs extrêmes)
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
        elif 0 < missing_perc <= 5:
            if col_type == "Numeric":
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")

    # === B. SUPPRESSION DES CONSTANTES ===
    # Deuxième passage de sécurité pour supprimer les colonnes à variance nulle
    constant_cols = [
        col for col, stats in variables.items()
        if col in df.columns and stats["n_distinct"] == 1
    ]
    df.drop(columns=constant_cols, inplace=True, errors="ignore")

    # === C. SUPPRESSION DES CORRÉLATIONS ÉLEVÉES ===
    # Si deux colonnes disent quasiment la même chose (corrélation > 0.85),
    # on en garde une seule pour éviter la redondance.
    correlations = desc.correlations
    if correlations and hasattr(correlations, "auto") and correlations.auto is not None:
        corr_matrix = correlations.auto.abs()
        cols_to_drop = set()

        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 == col2:
                    continue
                if corr_matrix.loc[col1, col2] > corr_threshold:
                    miss1 = variables[col1]["p_missing"]
                    miss2 = variables[col2]["p_missing"]
                    col_drop = col1 if miss1 >= miss2 else col2
                    cols_to_drop.add(col_drop)

        df.drop(columns=list(cols_to_drop), inplace=True, errors="ignore")

    return df