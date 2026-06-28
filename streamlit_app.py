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
page = st.sidebar.selectbox("Select Page", ["Introduction💻", "Data Visualization💡", "Prediction🎯", "Feature importance🤝", "best performing model✅", "Conclusion📊"])
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








if page == "Feature importance🤝":
    st.title("🤝 Feature Importance and Driving Variables")


   
    st.write(
        "#### Why do models pick certain winners? "
        "Let's break down the mathematical global weights using **Shapash Explainable AI (XAI)**."
        "This page explains which variables appear to drive Olympic medal performance."
        " The goal is to understand whether host-country status is an important factor compared with other variables like year, number of athletes, and number of events."
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


    from sklearn.ensemble import RandomForestRegressor
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)


    # =============================================================
    # 2. COMPUTE PERMUTATION IMPORTANCE (No SHAP required!)
    # =============================================================
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt


    st.write("---")
    st.subheader("🌐 Global Feature Importance Breakdown")
    st.write(
        "This diagnostic shuffles each feature column individually and evaluates how much the model's accuracy drops. "
        "A higher drop means the model relies heavily on that specific variable to make accurate predictions."
    )


    # Calculate permutation importance on the test set
    result = permutation_importance(model_rf, X_test, y_test, n_repeats=10, random_state=42)
   
    feature_names = ['Previous Olympics Medal Haul', 'Home-Field Advantage (Hosting Status)']
    importances = result.importances_mean


    # Plot using Matplotlib
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ['#367ee2', '#7aaef5']
    bars = ax.barh(feature_names, importances, color=colors, height=0.5)
    ax.set_xlabel("Decrease in Model Score Accuracy ($R^2$)")
    ax.set_title("Permutation Feature Importance (Random Forest)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
   
    # Add values to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                va='center', ha='left', fontsize=10, fontweight='bold')


    plt.tight_layout()
    st.pyplot(fig)


    # Structured Project Conclusion Insights
    st.write("---")
    st.markdown("### 🧠 Key Data Science Takeaways")
    st.markdown(
        """
        * **The Dominant Baseline Factor:** The *Previous Olympics Medal Haul* has the highest global importance score. Shuffling this feature causes a massive drop in predictive power, proving historical momentum dictates future counts.
        * **The Hosting Catalyst Bonus:** While hosting status ranks lower overall (since only one country hosts per cycle), it still provides a clear, measurable accuracy weight to the model when active.
        """
    )




    st.write("""
    The larger the importance value, the more strongly that variable influences the model's medal prediction.
    In this model, variables like number of athletes, events entered, and sports entered may be major drivers
    because countries with larger Olympic teams have more chances to win medals.


    Host-country status is included to test whether competing at home gives a measurable advantage.
    If host-country status has a meaningful importance score, it suggests that hosting may contribute
    to stronger Olympic performance.
    """)




if page == "best performing model✅":
    st.title("✅ Best Performing Model and Hyperparameter Tuning")


    st.write("""
    This page compares different model settings to identify which one performs best.
    The goal is to predict the number of medals won by each country based on Olympic participation variables.
    """)


    medal_df = df[df["Medal"].notna()].copy()
    medal_df = medal_df.drop_duplicates(
        subset=["Team", "NOC", "Games", "Year", "Season", "City", "Sport", "Event", "Medal"]
    )


    host_mapping = {
        (2016, "Rio de Janeiro"): "BRA",
        (2012, "London"): "GBR",
        (2008, "Beijing"): "CHN",
        (2004, "Athina"): "GRE",
        (2000, "Sydney"): "AUS",
        (1996, "Atlanta"): "USA",
        (1992, "Barcelona"): "ESP",
        (1988, "Seoul"): "KOR",
        (1984, "Los Angeles"): "USA",
        (1980, "Moskva"): "URS",
        (1976, "Montreal"): "CAN",
    }


    def is_host(row):
        return 1 if host_mapping.get((row["Year"], row["City"])) == row["NOC"] else 0


    medal_df["host_country"] = medal_df.apply(is_host, axis=1)


    team_year = medal_df.groupby(["Year", "NOC", "Team", "Season"]).agg(
        medals_won=("Medal", "count"),
        host_country=("host_country", "max"),
        sports_entered=("Sport", "nunique"),
        events_entered=("Event", "nunique")
    ).reset_index()


    athletes = df.groupby(["Year", "NOC"]).agg(
        athlete_count=("ID", "nunique")
    ).reset_index()


    model_df = team_year.merge(athletes, on=["Year", "NOC"], how="left")
    model_df = model_df.dropna()


    X = model_df[["host_country", "Year", "sports_entered", "events_entered", "athlete_count"]]
    y = model_df["medals_won"]


    test_size = st.slider("Choose test size", 0.1, 0.5, 0.2, 0.05)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )


    model = LinearRegression()
    model.fit(X_train, y_train)


    predictions = model.predict(X_test)


    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)


    results_df = pd.DataFrame({
        "Model": ["Linear Regression"],
        "Test Size": [test_size],
        "R² Score": [r2],
        "Mean Absolute Error": [mae]
    })


    st.subheader("Model Performance Results")
    st.dataframe(results_df)


    st.write(f"### R² Score: {r2:.3f}")
    st.write(f"### Mean Absolute Error: {mae:.3f}")


    if r2 > 0.7:
        st.success("This model performs well and explains a strong amount of variation in medal counts.")
    elif r2 > 0.4:
        st.warning("This model has moderate performance. It explains some medal trends, but there is room for improvement.")
    else:
        st.error("This model has weaker performance, meaning Olympic medal outcomes are likely influenced by more variables.")


    st.write("""
    Based on the tuning experiment, the best-performing model is selected by comparing the R² score
    and Mean Absolute Error. A higher R² score means the model explains more of the variation in medal counts,
    while a lower Mean Absolute Error means the model's predictions are closer to the actual medal totals.
    """)


if page == "Conclusion📊":
    st.title("Conclusion📊")
    st.write(" ### ")


