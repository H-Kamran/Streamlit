from xml.etree.ElementInclude import include
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image
import os
from sklearn.impute import SimpleImputer

icon = Image.open('images/icon.jpg')
logo = Image.open('images/logo.jpg')
banner = Image.open('images/banner.jpg')

st.set_page_config(layout='wide',
                   page_title='Python',
                   page_icon=icon)

st.title('Analysis and modeling of two dataset')
st.text('Machine Learning Web Application with Streamlit')

# Sidebar container
st.sidebar.image(image=logo)

menu = st.sidebar.selectbox('', ['Homepage', 'EDA', 'Modeling'])

if menu == 'Homepage':
    # Homepage container
    st.header('Homepage')
    st.image(banner, use_column_width='always')

    dataset = st.selectbox(
        'Select dataset', ['Loan Prediction', 'Water Potability'])
    st.markdown('Selected: **{0}** Dataset'.format(dataset))

    if dataset == 'Loan Prediction':
        st.warning('''
                    **The Problem**:
        Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
        Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form.
        It is a classification problem where we have to predict whether a loan would be approved or not.
        We'll start by exploratory data analysis, then preprocessing, and finally we'll be testing different models such as decision trees.
        The data consists of following rows:

        ''')
        st.info('''

        **Loan_ID** Unique Loan ID

        **Gender** Male/ Female

        **Married** Applicant married (Y/N)

        **Dependents** Number of dependents

        **Education** Applicant Education (Graduate/ Under Graduate)

        **Self_Employed** Self employed (Y/N)

        **ApplicantIncome** Applicant income

        **CoapplicantIncome** Coapplicant income

        **LoanAmount** Loan amount in thousands

        **Loan_Amount_Term** Term of loan in months

        **Credit_History** credit history meets guidelines

        **Property_Area** Urban/ Semi Urban/ Rural

        **Loan_Status** (Target) Loan approved (Y/N)

        ''')
    else:
        st.warning('''
                    **The Problem**:
        Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level.
        The data consists of following rows:

        ''')
        st.info('''

        **ph** pH of 1. water (0 to 14).

        **Hardness** Capacity of water to precipitate soap in mg/L.

        **Solids** Total dissolved solids in ppm.

        **Chloramines** Amount of Chloramines in ppm.

        **Sulfate** Amount of Sulfates dissolved in mg/L.

        **Conductivity** Electrical conductivity of water in μS/cm.

        **Organic_carbon** Amount of organic carbon in ppm.

        **Trihalomethanes** Amount of Trihalomethanes in μg/L.

        **Turbidity** Measure of light emiting property of water in NTU.

        **Potability** Indicates if water is safe for human consumption. Potable - 1 and Not potable - 0

        ''')


elif menu == 'EDA':
    def outlier_treatment(datacolumn):
        Q1, Q3 = np.percentile(datacolumn, [25, 75])
        IQR = Q3-Q1
        lower_range = Q1-(1.5*IQR)
        upper_range = Q3+(1.5*IQR)
        return lower_range, upper_range

    def describeStat(df):
        st.dataframe(df)
        st.subheader('Statistical Values')
        df.describe().T

        st.subheader('Balance of Data')
        st.bar_chart(df.iloc[:, -1].value_counts())

        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ['Columns', 'Counts']

        c_eda1, c_eda2, c_eda3 = st.columns(
            [2.5, 1.5, 2.5])  # approximately 38.5% 23% 38.5
        c_eda1.subheader('Null Variables')
        c_eda1.dataframe(null_df)

        c_eda2.subheader('Imputation')
        cat_method = c_eda2.radio('Categorical', ['Mode', 'Backfill', 'Ffill'])
        num_method = c_eda2.radio('Numerical', ['Mode', 'Median'])

        # Feature Engineering
        c_eda2.subheader('Feature Engineering')
        balance_problem = c_eda2.checkbox('Under Sampling')
        outlier_problem = c_eda2.checkbox('Clean Outlier')

        if c_eda2.button('Data preprocessing'):

            # Data Cleaning
            cat_array = df.iloc[:, :-1].select_dtypes(include='object').columns
            num_array = df.iloc[:, :-1].select_dtypes(exclude='object').columns

            if cat_array.size > 0:
                if cat_method == 'Mode':
                    imp_cat = SimpleImputer(
                        missing_values=np.nan, strategy='most_frequent')
                    df[cat_array] = imp_cat.fit_transform(df[cat_array])
                elif cat_method == 'Backfill':
                    df[cat_array].fillna(method='backfill', inplace=True)
                else:
                    df[cat_array].fillna(method='ffill', inplace=True)

            if num_array.size > 0:
                if num_method == 'Mode':
                    imp_num = SimpleImputer(
                        missing_values=np.nan, strategy='most_frequent')
                else:
                    imp_num = SimpleImputer(
                        missing_values=np.nan, strategy='median')
                df[num_array] = imp_num.fit_transform(df[num_array])

            df.dropna(axis=0, inplace=True)

            if balance_problem:
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler()
                X = df.iloc[:, :-1]
                Y = df.iloc[:, [-1]]
                X, Y = rus.fit_resample(X, Y)
                df = pd.concat([X, Y], axis=1)

            if outlier_problem:
                for col in num_array:
                    lowerbound, upperbound = outlier_treatment(df[col])
                    df[col] = np.clip(
                        df[col], a_min=lowerbound, a_max=upperbound)

            null_df = df.isnull().sum().to_frame().reset_index()
            null_df.columns = ['Columns', 'Counts']
            c_eda3.subheader('Null Variables')
            c_eda3.dataframe(null_df)
            c_eda2.subheader('Balance of Data')
            st.bar_chart(df.iloc[:, -1].value_counts())

            heatmap = px.imshow(df.corr())
            st.plotly_chart(heatmap)
            st.dataframe(df)

            if os.path.exists('formodel.csv'):
                os.remove('formodel.csv')
            df.to_csv('formodel.csv', index=False)

    # Homepage Container
    st.header('Exploratory Data Analysis')
    dataset = st.selectbox(
        'Select dataset', ['Loan Prediction', 'Water Potability'])

    if dataset == 'Loan Prediction':
        df = pd.read_csv('datasets/loan_prediction.csv')
        describeStat(df)
    else:
        df = pd.read_csv('datasets/water_potability.csv')
        describeStat(df)
else:
    # Modeling Container
    st.header('Modeling')
    if not os.path.exists('formodel.csv'):
        st.header('Please Run Preprocessing')
    else:
        df = pd.read_csv('formodel.csv')
        st.dataframe(df)

        c_model1, c_model2 = st.columns(2)

        c_model1.subheader('Scaling')
        scaling_method = c_model1.radio('', ['Standard', 'Robust', 'MinMax'])
        c_model2.subheader('Encoding')
        encoder_method = c_model2.radio('', ['Label', 'One-Hot'])

        st.header('Train and Test Splitting')
        c_model1_1, c_model2_1 = st.columns(2)
        random_state = c_model1_1.text_input('Random State')
        test_size = c_model2_1.text_input('Percentage')

        model = st.selectbox('Select Model', ['Xgboost', 'Catboost'])
        st.markdown('Selected: **{0}** Dataset'.format(model))

        if st.button('Run Model'):
            from sklearn.preprocessing import LabelEncoder
            lb =LabelEncoder()
            df['Loan_Status']=lb.fit_transform(df['Loan_Status'])

            cat_array = df.iloc[:, :-1].select_dtypes(include='object').columns
            num_array = df.iloc[:, :-1].select_dtypes(exclude='object').columns
            Y = df.iloc[:, [-1]]

            if num_array.size > 0:
                if scaling_method == 'Standard':
                    from sklearn.preprocessing import StandardScaler
                    sc = StandardScaler()
                elif scaling_method == 'Robust':
                    from sklearn.preprocessing import RobustScaler
                    sc = RobustScaler()
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    sc = MinMaxScaler()
                df[num_array] = sc.fit_transform(df[num_array])

            if cat_array.size > 0:
                if encoder_method == 'Label':
                    # from sklearn.preprocessing import LabelEncoder
                    # lb = LabelEncoder()
                    for col in cat_array:
                        df[col]=lb.fit_transform(df[col])
                else:
                    df.drop(df.iloc[:, [-1]], axis=1, inplace=True)
                    dms_df = df[cat_array]
                    dms_df = pd.get_dummies(dms_df, drop_first=True)
                    df_ = df.drop(cat_array, axis=1)
                    df = pd.concat([df_, dms_df, Y], axis=1)

            st.dataframe(df)
    # Modeling Part
            X = df.iloc[:, :-1]
            Y = df.iloc[:, [-1]]

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=int(test_size), random_state=int(random_state))

            st.markdown('X_train size = {0}'.format(X_train.shape))
            st.markdown('X_test size = {0}'.format(X_test.shape))
            st.markdown('y_train size = {0}'.format(y_train.shape))
            st.markdown('y_test size = {0}'.format(y_test.shape))

            # st.title('Congratulations Your Model is working')

            if model == 'Xgboost':
                import xgboost as xgb
                model = xgb.XGBClassifier().fit(X_train, y_train)
            else:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier().fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]

            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
            st.markdown('Confusion Matrix')
            st.write(confusion_matrix(y_test, y_pred))

            # creating dataframe from classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()

            st.dataframe(df_report)

            accuracy = str(round(accuracy_score(y_test, y_pred), 2))+'%'
            st.markdown('Accuracy Score = '+accuracy)

            # ROC Curve
            from sklearn.metrics import roc_curve, auc
            fpr,tpr,thresholds=roc_curve(y_test,y_score)
            fig=px.area(
                x=fpr,
                y=tpr,
                title='Roc Curve',
                labels=dict(x='False Positive Rate',
                            y='True Positive Rate'),
                width=700,
                height=700
            )
            fig.add_shape(
                type='line',
                line=dict(dash='dash'),
                x0=0,
                x1=1,
                y0=0,
                y1=1
            )

