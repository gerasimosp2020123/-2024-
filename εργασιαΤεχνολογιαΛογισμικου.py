import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import umap

# 1. Data Upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv", "xlsx", "tsv"])

df = None  # Initialize df to None

if uploaded_file is not None:
    # Load the data as before
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, delimiter='\t')
    
    st.write("Dataset Preview:", df)

    # Convert categorical variables to numerical using one-hot encoding
    df = pd.get_dummies(df, drop_first=True)  # drop_first=True to avoid dummy variable trap

    # 2. Visualization Tab
    st.sidebar.header("Visualization")
    
    if df is not None and not df.empty:
        if st.sidebar.button("2D PCA"):
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df.iloc[:, :-1])
            fig = px.scatter(pca_result, x=0, y=1, color=df.iloc[:, -1])
            st.plotly_chart(fig)

        if st.sidebar.button("3D PCA"):
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df.iloc[:, :-1])
            fig = px.scatter_3d(pca_result, x=0, y=1, z=2, color=df.iloc[:, -1])
            st.plotly_chart(fig)
        
        if st.sidebar.button("2D UMAP"):
            reducer = umap.UMAP(n_components=2)
            umap_result = reducer.fit_transform(df.iloc[:, :-1])
            fig = px.scatter(umap_result, x=0, y=1, color=df.iloc[:, -1])
            st.plotly_chart(fig)

        if st.sidebar.button("3D UMAP"):
            # Option 1: Reduce to 2D
            reducer = umap.UMAP(n_components=2)  # Change to 2 components
            umap_result = reducer.fit_transform(df.iloc[:, :-1])
            fig = px.scatter(umap_result, x=0, y=1, color=df.iloc[:, -1])
            st.plotly_chart(fig)

            # Option 2: Convert to dense array (if keeping 3D)
            # reducer = umap.UMAP(n_components=3)
            # umap_result = reducer.fit_transform(df.iloc[:, :-1].to_numpy())  # Convert to dense array
            # fig = px.scatter_3d(umap_result, x=0, y=1, z=2, color=df.iloc[:, -1])
            # st.plotly_chart(fig)
    
    else:
        st.warning("Please upload a valid dataset to visualize.")

    # EDA Plots
    st.sidebar.header("Exploratory Data Analysis")
    if st.sidebar.button("Show EDA Plots"):
        st.write("Histogram of Features")
        st.write(df.hist())
        st.write("Correlation Matrix")
        st.write(df.corr())

    # 3. Feature Selection Tab
    st.sidebar.header("Feature Selection")
    k = st.sidebar.slider("Number of features", min_value=1, max_value=len(df.columns)-1, value=5)
    if st.sidebar.button("Select Features"):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        selector = SelectKBest(chi2, k=k).fit(X, y)
        selected_features = X.columns[selector.get_support()]
        reduced_df = df[selected_features]
        st.write("Dataset with Selected Features:", reduced_df)

    # ... (previous code remains the same)

# ... (previous code remains the same)

# 4. Classification Tab
if df is not None and not df.empty:  # Check if df is defined and not empty
    st.sidebar.header("Classification")
    algorithm = st.sidebar.selectbox("Choose an Algorithm", ["k-Nearest Neighbors", "Random Forest"])

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.3)

    max_neighbors = min(len(X_train) - 1, 10)  # Set maximum number of neighbors

    if max_neighbors > 1:
        param = st.sidebar.slider("Parameter", min_value=1, max_value=max_neighbors, value=min(3, max_neighbors))
    else:
        param = 1
        st.sidebar.write("Dataset too small for parameter selection. Using default value of 1.")

    if algorithm == "k-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=param)
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=param)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Υπολογισμός μετρικών απόδοσης
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')  # Χρησιμοποιήστε 'macro' ή 'micro' αν έχετε πολλές κλάσεις
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if len(set(y_test)) == 2 else None

    # Εμφάνιση αποτελεσμάτων
    st.subheader("Αποτελέσματα Αλγορίθμου")
    st.write(f"Ακρίβεια: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    if roc_auc is not None:
        st.write(f"ROC AUC: {roc_auc:.2f}")
    else:
        st.write("ROC AUC: Δεν είναι διαθέσιμο (περισσότερες από 2 κλάσεις).")

# 5. Info Tab
st.sidebar.header("Info")
if st.sidebar.button("Show Info"):
    st.title("Πληροφορίες για την Εφαρμογή")
    st.write(""" 
        Αυτή η εφαρμογή έχει σχεδιαστεί για να αναλύει δεδομένα και να παρέχει οπτικοποιήσεις 
        και αποτελέσματα από διάφορους αλγορίθμους μηχανικής μάθησης.
    """)
    
    st.subheader("Πώς Λειτουργεί")
    st.write(""" 
        1. **Φόρτωση Δεδομένων:** Ο χρήστης μπορεί να ανεβάσει ένα αρχείο CSV, Excel ή TSV.
        2. **Οπτικοποιήσεις:** Η εφαρμογή παρέχει οπτικοποιήσεις μέσω PCA και UMAP.
        3. **Επιλογή Χαρακτηριστικών:** Ο χρήστης μπορεί να επιλέξει τα χαρακτηριστικά που θέλει να χρησιμοποιήσει.
        4. **Κατηγοριοποίηση:** Η εφαρμογή εκπαιδεύει μοντέλα και παρέχει μετρικές απόδοσης.
    """)

    st.subheader("Ομάδα Ανάπτυξης")
    st.write(""" 
        - **Νικόλαος-Σπυρίδων Καλιβόπουλος** - Αριθμός Μητρώου: Π2020016
        - **Γεράσιμος Τσιλιμπάρης** - Αριθμός Μητρώου: Π2020123
    """)

    st.subheader("Tasks")
    st.write(""" 
        - **Νικόλαος-Σπυρίδων Καλιβόπουλος:** 
            - Task 1: ΕΡΑΓΑΣΤΗΚΑΜΕ ΚΑΙ ΟΙ ΔΥΟ ΣΕ ΟΛΑ ΤΑ TASKS ΤΗΣ ΕΡΓΑΣΙΑΣ
        - **Γεράσιμος Τσιλιμπάρης:** 
            - Task 1: ΕΡΑΓΑΣΤΗΚΑΜΕ ΚΑΙ ΟΙ ΔΥΟ ΣΕ ΟΛΑ ΤΑ TASKS ΤΗΣ ΕΡΓΑΣΙΑΣ
    """)