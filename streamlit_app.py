import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from shapash import SmartExplainer
import wandb


## logo and url name
st.set_page_config(
    page_title="The Olympic Games🌎",
    layout="wide",
    page_icon="🌎",
)

CSV_URL = "https://docs.google.com/spreadsheets/d/1E-64FAAWIx8apAY1mFqt5hwb_xRAOyG-PhUhDFL2BBY/export?format=csv"
@st.cache_data
def get_olympic_data():
    return pd.read_csv(CSV_URL, encoding="latin-1", thousands=",")

df = get_olympic_data()


##title page
st.markdown("<h1 style='text-align: center; color: white;'>🏆 The Olympic Games Explorer🌎</h1>", unsafe_allow_html=True)
page = st.sidebar.selectbox("Select Page", ["Introduction💻", "Data Visualization💡", "Prediction🎯", "Explainability🤝", "Best performing model✅", "Conclusion📊"])
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images.png")


## change the font to Times New Roman
st.markdown("""
<style>
    /* Target every text-bearing container, header, input, and paragraph */
    html, body, [data-testid="stAppViewContainer"], .stApp,
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, a {
        font-family: 'Times New Roman', Times, serif !important;
    }
</style>
""", unsafe_allow_html=True)


## making sidebar dark blue and background blue
st.markdown("""<style>[data-testid="stSidebar"] { background-color: #7aaef5;} </style>""", unsafe_allow_html=True)
st.markdown(""" <style> .stApp { background-color: #c7e4ff;} </style> """, unsafe_allow_html=True)
colorX = ["#7aaef5"]
colorY = ["#c7e4ff"]  


## introduction page
if page == "Introduction💻":
    st.title("🤿A deep dive into the Olympic Games")
    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;">Business Case Presentation🔎</span>', unsafe_allow_html=True)
   
    ## Project overview
    st.write(""" #### The Olympic Games are among the biggest sporting events on Earth.
    They feature summer and winter sports events, each held once every 4 years,
    in which thousands of athletes from around the world compete in a variety of athletic events.
    The Olympic Games, open to both amateur and professional athletes, involve more than 200 teams,
    each team representing a sovereign state or territory.
     
    The International Olympic Committee (IOC) selects host cities by evaluating interested regions on
    sustainability, infrastructure, and public support, with the process overseen by permanent Future
    Host Commissions. The IOC Executive Board puts forward the preferred host for a final vote.
    The full committee of IOC members then votes to officially elect the host city for the next Olympic Games.
               
    #### Research Question
    Does hosting the Olympics improve performance? Is this an unfair advantage? Who should host the next Olympics?


    #### Objectives
    - By evaluating data from every Olympics up to Tokyo 2020, we will determine if the athletes from the
    hosting country produce more medals while competing at home. """)


    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;">Data Presentation🗂️</span>', unsafe_allow_html=True)
   
    st.write(""" #### This historical dataset on the modern Olympic Games, including all the Games from Athens 1896
    to Tokyo 2020. The columns include the athletes ID number, name, sex, height, weight, team (country),
    National Olympic Committee 3-letter code, The games year and season, the host city, the sport,
    the event, and the medal earned. There are 271116 rows and 15 columns.  """)


    st.write(""" #### Here is an example of the first 10 inputs from the Athens 1896 Olympic games to see the layout and columns provided.""")
    first_10_rows = df.head(10)
    st.dataframe(first_10_rows)


## data viz page
if page == "Data Visualization💡":
    ## pick out event, gender, and year and see dataset print out
    st.title("👏 Olympic Data Finder")
    st.write(" #### Select filters to extract exact rows from the dataset")
   
    selected_event = st.selectbox("Pick the Event:", sorted(df['Event'].unique()))
    selected_year = st.selectbox("Pick the Year:", sorted(df['Year'].unique(), reverse=True))
   
    filtered_df = df[(df['Event'] == selected_event)  & (df['Year'] == selected_year)]


    st.write("### Matching Results:")
    if not filtered_df.empty:
        st.dataframe(filtered_df)
    else:
        st.warning("No data found matching that exact combination")


    ## summer events vs. winter events pie chart
    st.title("❄️ vs ☀️ Olympic Events Distribution")
    events_per_season = df.groupby('Season')['Event'].nunique()
    fig, ax = plt.subplots(figsize=(3, 3))
    colors = ["#f8f434", "#367ee2"]
   
    ax.pie(
        events_per_season,
        labels=events_per_season.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12}
    )
    ax.set_title('Summer vs Winter Olympics Event Breakdown', fontsize=14, pad=20)
    st.pyplot(fig)


    ## seeing medal count at 2016 for top 15 countries at rio olympics
    st.title("Medal Breakdown🥇")
    st.write("#### A quick glimpse at the top 15 scorers at the 2016 Rio Olympics")
    rio_2016 = df[(df['Year'] == 2016) & (df['Season'] == 'Summer')].copy()
    rio_unique = rio_2016.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    medal_winners = rio_unique[rio_unique['Medal'].notna()]
    medal_counts = pd.crosstab(index=medal_winners['Team'], columns=medal_winners['Medal']).reset_index()


    medal_counts = medal_counts.rename(columns={
        'Team': 'Team/NOC',
        'Gold': 'Gold Medal',
        'Silver': 'Silver Medal',
        'Bronze': 'Bronze Medal'
    })
    medal_counts['Total'] = medal_counts['Gold Medal'] + medal_counts['Silver Medal'] + medal_counts['Bronze Medal']


    df_21_full = medal_counts.sort_values(by=['Gold Medal', 'Silver Medal', 'Bronze Medal'], ascending=False).reset_index(drop=True)
    df_21_full.insert(0, 'Rank', df_21_full.index + 1)
    top_15_countries = df_21_full[['Rank', 'Team/NOC', 'Bronze Medal', 'Silver Medal', 'Gold Medal', 'Total']].iloc[:15]
    st.dataframe(top_15_countries)
    st.write("#### As you can see Brazil (the host country) finished in 14th place")


##Prediction page
if page == "Prediction🎯":
    st.title("🌏Does hosting give you an advantage?")
    st.write("#### First, lets break down by Medals Won by Host Country vs. All Other Countries")


    medal_df = df[df['Medal'].notna()].copy()
    medal_df = medal_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal'])


    # 3. Create a dictionary mapping specific Cities/Years to their host NOC
    host_mapping = {
        (2016, 'Rio de Janeiro'): 'BRA',
        (2012, 'London'): 'GBR',
        (2008, 'Beijing'): 'CHN',
        (2004, 'Athina'): 'GRE',
        (2000, 'Sydney'): 'AUS',
        (1996, 'Atlanta'): 'USA',
        (1992, 'Barcelona'): 'ESP',
        (1988, 'Seoul'): 'KOR',
        (1984, 'Los Angeles'): 'USA',
        (1980, 'Moskva'): 'URS',
        (1976, 'Montreal'): 'CAN',
    }


   
    def is_host(row):
        if host_mapping.get((row['Year'], row['City'])) == row['NOC']:
            return "Host Country"
        return "Other Countries"


    medal_df['Host_Status'] = medal_df.apply(is_host, axis=1)


    # 5. Group data by Year and Host Status for Summer games
    summer_medals = medal_df[medal_df['Season'] == 'Summer']
    graph_data = summer_medals.groupby(['Year', 'Host_Status']).size().unstack(fill_value=0).reset_index()


    # Filter to look at modern games (e.g., 1976 onwards)
    graph_data = graph_data[graph_data['Year'] >= 1976]


    # 6. Plot the graph using Streamlit's built-in bar chart
    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;"> 1. Total Medals Distributed per Olympic Year</span>', unsafe_allow_html=True)
    chart_df = graph_data.set_index('Year')[['Host Country', 'Other Countries']]
   
    st.bar_chart(
        data=graph_data,
        x="Year",
        y=["Host Country", "Other Countries"],
        color=["#367ee2", "#7aaef5"],
        x_label="Year",
        y_label="Medal Count"
    )


    st.info(" Each bar represents the TOTAL number of medals at each Olympics. The dark blue slice represents medals won by the the Host Country. 💡You can see how prominent their home-field advantage can be relative to historical baselines.")


    ##part two, find average of how MUCH better they preform
    st.write("#### Next, lets use previous Olympic medal counts to figure out just how much better a country does when they are hosting")
    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;"> 2. Calculating the Historical Hosting Advantage</span>', unsafe_allow_html=True)


   
    simple_medals = summer_medals.groupby(['Year', 'City', 'NOC']).size().reset_index(name='Medal_Count')
    simple_medals['Is_Host'] = simple_medals.apply(lambda r: 1 if host_mapping.get((r['Year'], r['City'])) == r['NOC'] else 0, axis=1)


    # Extract averages for countries when they host vs when they don't
    host_nocs = list(host_mapping.values())
    host_history = simple_medals[simple_medals['NOC'].isin(host_nocs)]
   
    avg_hosting = host_history[host_history['Is_Host'] == 1]['Medal_Count'].mean()
    avg_not_hosting = host_history[host_history['Is_Host'] == 0]['Medal_Count'].mean()
    hosting_bump = avg_hosting - avg_not_hosting


    # Display clean metrics cards
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(label="Avg Medals (While Hosting)", value=f"{avg_hosting:.1f}")
    with col_m2:
        st.metric(label="Avg Medals (When Abroad)", value=f"{avg_not_hosting:.1f}")
    with col_m3:
        st.metric(label="The Hosting Performance Bump", value=f"+{hosting_bump:.1f}", delta=f"{hosting_bump:.1f} medals")
   
    st.info("By counting the avergae number of medals won and what percentage of those medals were from a hosting team, we are able to calculate just how much better a team preforms when they are at home")


    ##PART3 LINEAR REGRESSION
    st.write("#### Now, let's train two separate Predictive Models to tackle the problem. One Linear Regression and one Random Forest. To make these models we used previous medal counts and host preformance data.")
    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;">3. Prediction Models</span>', unsafe_allow_html=True)


    ml_data = simple_medals.sort_values(by=['NOC', 'Year'])
    ml_data['Prev_Medal_Count'] = ml_data.groupby('NOC')['Medal_Count'].shift(1)
    ml_data = ml_data.dropna()


    X = ml_data[['Prev_Medal_Count', 'Is_Host']]
    y = ml_data['Medal_Count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    pred_lr = model_lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, pred_lr)
    r2_lr = r2_score(y_test, pred_lr)


    # Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    r2_rf = r2_score(y_test, pred_rf)


    #results side-by-side
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.write("**Model 1: Linear Regression (Baseline)**")
        st.write(f"- Mean Absolute Error (MAE): `{mae_lr:.2f}` medals")
        st.write(f"- R² Accuracy Score: `{r2_lr:.3f}`")
    with col_v2:
        st.write("**Model 2: Random Forest (Advanced Ensemble)**")
        st.write(f"- Mean Absolute Error (MAE): `{mae_rf:.2f}` medals")
        st.write(f"- R² Accuracy Score: `{r2_rf:.3f}`")


    st.write("#### 🔮Live Medal Predictor Simulator")
   
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        user_prev_medals = st.number_input("How many medals did they win at the LAST Olympics?", min_value=0, max_value=150, value=20, step=1)
    with col_in2:
        user_host = st.selectbox("Is this country hosting the CURRENT games?", ["No, competing abroad", "Yes, hosting at home"])
    with col_in3:
        chosen_model = st.selectbox("Which Model should run the prediction?", ["Model 1: Linear Regression", "Model 2: Random Forest"])


   
    host_binary = 1 if user_host == "Yes, hosting at home" else 0
    input_features = np.array([[user_prev_medals, host_binary]])


    if chosen_model == "Model 1: Linear Regression":
        final_prediction = model_lr.predict(input_features)[0]
    else:
        final_prediction = model_rf.predict(input_features)[0]


    final_output = max(0, round(final_prediction))


    # Display dynamic calculator window box
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 8px; border: 2px solid #367ee2; text-align: center; margin-top: 15px;">
        <h4 style="color: black; margin: 0;">Predicted Total Medal Output ({chosen_model.split(':')[1].strip()}):</h4>
        <h1 style="color: #367ee2; font-size: 50px; margin: 10px 0;">🏆 {final_output} Medals</h1>
    </div>
    """, unsafe_allow_html=True)








if page == "Explainability🤝":
    st.title("🤝 Feature Importance and Driving Variables")
    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;">🔍 AI Explainability with Shapash</span>', unsafe_allow_html=True)
    st.write(
        "While standard feature diagnostics tell us which variables matter overall, Shapash Explainable AI (XAI) "
        "calculates the exact mathematical contribution of each feature. This helps us understand precisely "
        "how our Random Forest and linear regression models make individual predictions."
    )
    medal_df = df[df['Medal'].notna()].copy()
    medal_df = medal_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal'])

    host_mapping = {
        (2016, 'Rio de Janeiro'): 'BRA', (2012, 'London'): 'GBR', (2008, 'Beijing'): 'CHN',
        (2004, 'Athina'): 'GRE', (2000, 'Sydney'): 'AUS', (1996, 'Atlanta'): 'USA',
        (1992, 'Barcelona'): 'ESP', (1988, 'Seoul'): 'KOR', (1984, 'Los Angeles'): 'USA',
        (1980, 'Moskva'): 'URS', (1976, 'Montreal'): 'CAN',
    }

    def is_host(row):
        return 1 if host_mapping.get((row['Year'], row['City'])) == row['NOC'] else 0

    medal_df['Is_Host'] = medal_df.apply(is_host, axis=1)
    summer_medals = medal_df[medal_df['Season'] == 'Summer']
    simple_medals = summer_medals.groupby(['Year', 'City', 'NOC', 'Is_Host']).size().reset_index(name='Medal_Count')
    
    ml_data = simple_medals.sort_values(by=['NOC', 'Year'])
    ml_data['Prev_Medal_Count'] = ml_data.groupby('NOC')['Medal_Count'].shift(1)
    ml_data = ml_data.dropna()

    X = ml_data[['Prev_Medal_Count', 'Is_Host']]
    y = ml_data['Medal_Count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Retrain the baseline Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    # 3. Setup and compile Shapash SmartExplainer Sandbox
    xpl = SmartExplainer(
        model=model_rf,
        features_dict={
            'Prev_Medal_Count': 'Previous Olympics Medal Haul',
            'Is_Host': 'Hosting Status'
        }
    )
    xpl.compile(x=X_test, y_target=y_test)


    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;">🌐Feature Significance</span>', unsafe_allow_html=True)
    st.write("This interactive chart breaks down how heavily the model relies on each variable across the entire dataset.")
    fig1 = xpl.plot.features_importance()
    st.plotly_chart(fig1, use_container_width=True)
    st.info("As you can see, the two most important driving variables to predict a countries medal count for a future Olympic are their medal count in the previous Olympics and if they are hosting or not.")

    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;">📈 Local Feature Contribution Dynamics</span>', unsafe_allow_html=True)
    st.write(
        "This scatter plot highlights the exact mathematical trend: as a country's historical "
        "medal haul increases, look at how sharply the model scales up its future predictions."
    )
    fig2 = xpl.plot.contribution_plot('Prev_Medal_Count')
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        """
        * **Driving variable:** The *Previous Olympics Medal Haul* exhibits dominant predictive weights. A country's number of athletes and success in certain events is the primary driver of future success.
        * **Hosting Bonus:** Even though hosting status acts on a much narrower subset of rows (since only one team hosts per cycle), Shapash isolated a visible baseline acceleration effect when `Home-Field Advantage` changes from 0 to 1.
        """
    )



if page == "Best performing model✅":
    st.title("✅ Best Performing Model and Hyperparameter Tuning")

    host_mapping = {
        (2016, 'Rio de Janeiro'): 'BRA', (2012, 'London'): 'GBR', (2008, 'Beijing'): 'CHN',
        (2004, 'Athina'): 'GRE', (2000, 'Sydney'): 'AUS', (1996, 'Atlanta'): 'USA',
        (1992, 'Barcelona'): 'ESP', (1988, 'Seoul'): 'KOR', (1984, 'Los Angeles'): 'USA',
        (1980, 'Moskva'): 'URS', (1976, 'Montreal'): 'CAN',
    }

    st.markdown('### <span style="background-color: white; color: black; padding: 2px 8px; border-radius: 4px;">Track, optimize, and lock in the best-performing predictive configuration.</span>', unsafe_allow_html=True)
    wandb_key = st.text_input("Enter W&B API Key", type="password")

    
    medal_df = df[df["Medal"].notna()].copy()
    medal_df = medal_df.drop_duplicates(subset=["Team", "NOC", "Games", "Year", "Season", "City", "Sport", "Event", "Medal"])
    
    def is_host(row):
        return 1 if host_mapping.get((row["Year"], row["City"])) == row["NOC"] else 0
    
    medal_df["host_country"] = medal_df.apply(is_host, axis=1)
    team_year = medal_df.groupby(["Year", "NOC", "Team", "Season"]).agg(
        medals_won=("Medal", "count"),
        host_country=("host_country", "max"),
        sports_entered=("Sport", "nunique"),
        events_entered=("Event", "nunique")
    ).reset_index()
    
    athletes = df.groupby(["Year", "NOC"]).agg(athlete_count=("ID", "nunique")).reset_index()
    model_df = team_year.merge(athletes, on=["Year", "NOC"], how="left").dropna()

    X = model_df[["host_country", "Year", "sports_entered", "events_entered", "athlete_count"]]
    y = model_df["medals_won"]

    st.markdown("### 🎛️ Tweak Random Forest Hyperparameters")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        n_estimators = st.slider("Number of Trees (n_estimators)", 10, 200, 100, 10)
    with col_t2:
        max_depth = st.slider("Max Tree Depth (max_depth)", 2, 20, 10, 1)
    with col_t3:
        test_size = st.slider("Validation Test Split Size", 0.1, 0.5, 0.2, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if st.button("🚀 Run Tuning Experiment & Log Metrics"):
        if not wandb_key:
            st.warning("⚠️ Please provide a W&B API Key in the sidebar to sync with the cloud board! Running local tracking baseline instead.")
        else:
            wandb.login(key=wandb_key)
            wandb.init(project="olympic-medal-predictor", config={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "test_size": test_size,
                "model_type": "RandomForestRegressor"
            })

        tuned_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        tuned_rf.fit(X_train, y_train)
        preds = tuned_rf.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        if wandb_key:
            wandb.log({"R2_Score": r2, "MAE": mae})
            wandb.finish()
            st.success("🎉 Run successfully sent to your Weights & Biases Dashboard project board!")

        st.write("---")
        st.subheader("📊 Current Experiment Yield Outcomes")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(label="R² Explanatory Variance Score", value=f"{r2:.4f}")
        with col_r2:
            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f} Medals")

        if r2 > 0.75:
            st.success("🔥 Outstanding Performance configuration profile!")
        else:
            st.info("💡 Tip: Try dialing up the Number of Trees or extending Max Depth properties to improve data feature fitting variance.")
    st.write(""" #### Ultimately, this page is finding the "sweet spot" to accurately forecast Olympic success. By adjusting the sliders above, you are controlling exactly how the Random Forest algorithm analyzes historical Olympic data to predict a country's total Medal Count.""")
    
if page == "Conclusion📊":

    st.title("📊 Conclusion")

    st.markdown(
        '<h3 style="background-color:white; color:black; padding:8px; border-radius:5px;">Final Project Summary</h3>',
        unsafe_allow_html=True
    )

    st.write("""
    Throughout this project we analyzed more than 100 years of Olympic history to determine
    whether hosting the Olympic Games gives countries a competitive advantage.

    By combining historical data, machine learning, explainable AI, and hyperparameter tuning,
    we were able to better understand the factors that influence Olympic medal success.
    """)

    st.divider()

    st.subheader("🏆 Key Findings")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Main Predictor", "Previous Medal Count")

    with col2:
        st.metric("Hosting Effect", "Positive Advantage")

    with col3:
        st.metric("Best Model", "Random Forest")

    st.info("""
    Countries that performed well in the previous Olympics were the most likely to perform
    well again. Hosting the Olympics also provided a measurable increase in medal production,
    although its impact was smaller than historical performance.
    """)

    st.divider()

    st.subheader("🤖 Machine Learning Summary")

    summary = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "Strengths": [
            "Simple, interpretable baseline",
            "Captures complex relationships"
        ],
        "Overall Performance": [
            "Good",
            "Best"
        ]
    })

    st.dataframe(summary, use_container_width=True)

    st.divider()

    st.subheader("💡 Future Improvements")

    st.write("""
    Future versions of this project could improve prediction accuracy by including:

    • GDP and country population

    • Number of athletes competing

    • Athlete world rankings

    • Olympic funding and investment

    • Home crowd attendance

    • Qualification statistics

    • More advanced machine learning models such as XGBoost
    """)

    st.divider()

    st.success(
        "Overall, our analysis supports the idea that hosting the Olympic Games provides a competitive advantage, while previous Olympic success remains the strongest predictor of future medal performance."
    )

    st.balloons()

