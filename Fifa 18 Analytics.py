

import numpy as np
import pandas as pd
from collections import Counter as counter
import matplotlib.pyplot as plt
import seaborn as sns





df = pd.read_csv("data.csv")

print("The columns of the dataframe are:")
print(df.columns)
print("DataFrame Description:")
print(df.describe())

print("Maximum Players are from the following countries:")
print(df['Nationality'].value_counts().head(10))


print("Maximum Ages of the Players:")
df.Age.value_counts().head(10)


print("DataFrame Shape:")
print(df.shape)

print("DataFrame Information:")
df.info()




# Selecting the columns we will work with
chosen_columns = [
    'Name',
    'Age',
    'Nationality',
    'Overall',
    'Potential',
    'Club',
    'Special',
    'Preferred Foot',
    'International Reputation',
    'Weak Foot',
    'Skill Moves',
    'Work Rate',
    'Position',
    'Height',
    'Weight',
    'Crossing',
    'Finishing', 
    'HeadingAccuracy', 
    'ShortPassing', 
    'Volleys', 
    'Dribbling',
    'Curve', 
    'FKAccuracy', 
    'LongPassing', 
    'BallControl', 
    'Acceleration',
    'SprintSpeed', 
    'Agility', 
    'Reactions', 
    'Balance', 
    'ShotPower',
    'Jumping', 
    'Stamina', 
    'Strength', 
    'LongShots', 
    'Aggression',
    'Interceptions', 
    'Positioning', 
    'Vision', 
    'Penalties', 
    'Composure',
    'Marking', 
    'StandingTackle', 
    'SlidingTackle', 
    'GKDiving', 
    'GKHandling',
    'GKKicking', 
    'GKPositioning', 
    'GKReflexes']


df_fifa19 = pd.DataFrame(df, columns = chosen_columns)



print("Let us find the null values in the dataset:")
print(df_fifa19.isnull().sum())


print("Let us remove some of the null values in the dataset:")
df_fifa19["Club"] = df_fifa19["Club"].fillna("Free Transfer")
df_fifa19["Preferred Foot"] = df_fifa19["Preferred Foot"].fillna("Unknown")
df_fifa19["International Reputation"] = df_fifa19["International Reputation"].fillna(0)
df_fifa19["Weak Foot"] = df_fifa19["Weak Foot"].fillna("0")
df_fifa19["Skill Moves"] = df_fifa19["Skill Moves"].fillna(0)
df_fifa19["Work Rate"] = df_fifa19["Work Rate"].fillna("Unknown")
df_fifa19["Position"] = df_fifa19["Position"].fillna("Unknown")
df_fifa19["Height"] = df_fifa19["Height"].fillna(0)
df_fifa19["Weight"] = df_fifa19["Weight"].fillna(0)

print("Let us see some of the changed null values in the dataset:")
print(df_fifa19.isnull().sum())

print(" Highest Overall Top 10 : ")
top10_Overall = df_fifa19.sort_values("Overall", ascending=False)[["Name", "Overall", "Age", "Nationality", "Club"]].head(10)
top10_Overall.set_index("Name", inplace=True)
print(top10_Overall)


x = np.array(df_fifa19.Overall)
plt.hist(x, bins = 70)
plt.title("Overall of Players vs Number of Players")
plt.xlabel("Overall")
plt.ylabel("Frequency")
plt.show()


x = np.array(df_fifa19.Age)
plt.hist(x, bins = 58)
plt.xlabel("Age")
plt.ylabel("Number of players")
plt.title("Players age distribution", fontsize=16)
plt.show()



oldest_players = df_fifa19.sort_values("Age", ascending=False)[["Name", "Nationality", "Club", "Position", "Age"]].head(10)
oldest_players.set_index("Name", inplace=True)
print(oldest_players)



youngest_players = df_fifa19.sort_values("Age")[["Name", "Nationality", "Club", "Position", "Age"]].head(10)
youngest_players.set_index("Name", inplace=True)
print(youngest_players)



selected_clubs = ("Liverpool", "Manchester City", "Chelsea", "Manchester United", "Tottenham", "Arsenal")

top6_clubs_age = df_fifa19[df_fifa19["Club"].isin(selected_clubs) & df_fifa19["Age"]]


sns.violinplot(x="Club", y="Age", data=top6_clubs_age)
sns.set_context("paper")
plt.show()


top6_clubs_potential = df_fifa19[df_fifa19["Club"].isin(selected_clubs) & df_fifa19["Potential"]]

sns.violinplot(x="Club", y="Potential", data=top6_clubs_potential)
sns.set_context("paper")
plt.show()

x_finishing = df_fifa19["Finishing"]
y_composure = df_fifa19["Composure"]
y_positioning = df_fifa19["Positioning"]
y_shotpower = df_fifa19["ShotPower"]
y_longshot = df_fifa19["LongShots"]

f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
ax1.scatter(x_finishing, y_composure, s=23)
ax1.set_xlabel("Finishing")
ax1.set_ylabel("Composure")

ax2.scatter(x_finishing, y_positioning, s=23)
ax2.set_xlabel("Finishing")
ax2.set_ylabel("Positioning")

ax3.scatter(x_finishing, y_shotpower, s=23)
ax3.set_xlabel("Finishing")
ax3.set_ylabel("Shot Power")

ax4.scatter(x_finishing, y_longshot, s=23)
ax4.set_xlabel("Finishing")
ax4.set_ylabel("Long Shot")

plt.subplots_adjust(top=0.5, right=0.8)

plt.show()



x_SprintSpeed = df_fifa19["SprintSpeed"]
y_acceleration = df_fifa19["Acceleration"]
y_agility = df_fifa19["Agility"]
y_balance = df_fifa19["Balance"]
y_dribbling = df_fifa19["Dribbling"]


f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
ax1.scatter(x_SprintSpeed, y_acceleration, s=23)
ax1.set_xlabel("Sprint Speed")
ax1.set_ylabel("Acceleration")

ax2.scatter(x_SprintSpeed, y_agility, s=23)
ax2.set_xlabel("Sprint Speed")
ax2.set_ylabel("Agility")

ax3.scatter(x_SprintSpeed, y_balance, s=23)
ax3.set_xlabel("Sprint Speed")
ax3.set_ylabel("Balance")

ax4.scatter(x_SprintSpeed, y_dribbling, s=23)
ax4.set_xlabel("Sprint Speed")
ax4.set_ylabel("Dribbling")

plt.subplots_adjust(top=0.5, right=0.8)

plt.show()



p = sns.countplot(x='Preferred Foot', data=df)
plt.show()

p = sns.countplot(x='Weak Foot', data=df)
plt.show()


p = sns.countplot(x='Position', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()




top_10 = df.head(10)
p = sns.barplot(x='Name', y='Finishing', data=top_10)
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()



plt.figure(1 , figsize = (15 , 7))
countries = []
c = counter(df['Nationality']).most_common()[:11]
for n in range(11):
    countries.append(c[n][0])

sns.countplot(x  = 'Nationality' ,
              data = df[df['Nationality'].isin(countries)] ,
              order  = df[df['Nationality'].isin(countries)]['Nationality'].value_counts().index , 
             palette = 'rocket') 
plt.xticks(rotation = 90)
plt.title('Maximum number footballers belong to which country' )
plt.show()


plt.figure(1 , figsize = (15 , 6))
sns.regplot(df['Age'] , df['Overall'])
plt.title('Scatter Plot of Age vs Overall rating')
plt.show()


df.sort_values(by = 'LongPassing', ascending = False)[['Name' , 'Club' , 'Nationality' , 
                                                     'Overall' , 'Value' , 'Wage']].head(5)



df.sort_values(by = 'ShotPower' , ascending = False)[['Name' , 'Club' , 'Nationality' , 
                                                     'ShotPower' ]].head(5)



import matplotlib.pyplot as plt
age = df.sort_values("Age")['Age'].unique()
reputation = df.groupby(by="Age")["International Reputation"].mean().values
plt.title("Age vs International Reputation")
plt.xlabel("Age")
plt.ylabel("International Reputation")
plt.plot(age, reputation)
plt.show()



import re
def convert_value(value):
    numeric_val = float(re.findall('\d+\.*\d*', value)[0])
    if 'M' in value:
        numeric_val*= 1000000
    elif 'K' in value:
        numeric_val*= 1000
    return int(numeric_val)
    


#Wage contains data in these formats(€0 ,€100k...)
df['Wage'] = df['Wage'].apply(lambda x: int(re.findall('\d+', x)[0])* 1000)
df['Value'] = df['Value'].apply(convert_value)




def plot_bar_plot(x = None, y = None, data = None ,hue =None, x_tick_rotation = None ,xlabel = None , 
                  ylabel = None , title = '',ylim = None, palette= None):
    plt.figure(1 , figsize = (15 , 6))
    if x_tick_rotation:
        plt.xticks(rotation = x_tick_rotation)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    sns.barplot(x = x, y = y, hue = hue, data = data, palette= palette)
    
top_30_players = df.sort_values(by = 'Overall' ,ascending = False ).head(30)
plot_bar_plot(x = "Name", y = "Overall", data = top_30_players, palette = 'RdBu', x_tick_rotation= 90 , ylim=(85,96) , title='Top 30 players with highest rating')
plt.show()



top_100_players = df.sort_values(by = 'Overall' ,ascending = False ).head(10)
plt.figure(1 , figsize = (15 , 6))
top_100_players['Age'].plot(kind = 'hist' , bins = 50)
plt.xlabel('Player\'s age')
plt.ylabel('Number of players')
plt.title('Top 10 players Age distribution')
plt.show()


print("Oldest Player in the top 100 teams is : ")


oldest_player_in_top_100 = top_100_players.sort_values(by = 'Age' ,ascending = False ).head(1)[['Name','Age','Overall','Club','Position']]
print("Oldest Player in the top 100 teams is : ")
print(oldest_player_in_top_100)





youngest_player_in_top_100 = top_100_players.sort_values(by = 'Age').head(1)[['Name','Age','Overall','Club','Position']]
print("Youngest Player in the top 100 teams is : ")
print(youngest_player_in_top_100)







df_g = df.groupby(['Club']).sum()[['Overall','Value','Wage']]
df_top_10_club_value = df_g.sort_values(by='Value',ascending=False).head(10)
df_top_10_club_value.reset_index(inplace=True)
print(" Highest Club Value : ")
print(df_top_10_club_value)



plot_bar_plot(x = "Club", y = "Value", data = df_top_10_club_value, x_tick_rotation= 90 , ylim=(5e8,9e8), palette="Blues_d", title="Club with highest player value")
plt.show()


print(" Highest Club Wages : ")
df_top_10_club_wage = df_g.sort_values(by='Wage',ascending=False).head(10)
df_top_10_club_wage.reset_index(inplace=True)
print(df_top_10_club_wage)




plot_bar_plot(x = "Club", y = "Wage", data = df_top_10_club_wage, x_tick_rotation= 90 , ylim=(1e6,5.5e6), palette="Blues_d", title="Club with highest player Wages")
plt.show()




df['difference'] = (df['Potential'] - (df['Overall']))




def evolution(d):
    if d == 0:
        return "Stable"
    elif d >=1 and d<=5:
        return "Small"
    elif d >=6 and d<=10:
        return "Medium"
    elif d >11:
        return "Big"





df['Evolution'] = df['difference'].apply(evolution)


print(" Highest Growth and Potential for Evolution : ")
print(df.loc[(df['Evolution']== 'Big') & (df['Potential']>80)].sort_values(by='Potential', ascending=False).head(10)[['Name','Age','Overall','Club','Position']])







plt.figure(figsize=(16,8))
sns.set_style("whitegrid")
plt.title('Grouping players by Age', fontsize=30, fontweight='bold', y=1.05,)
plt.xlabel('Number of players', fontsize=25)
plt.ylabel('Players Age', fontsize=25)
sns.countplot(x="Age",data = df, palette="hls");
plt.show()





plt.figure(figsize=(16,8))
sns.set_style("whitegrid")
plt.title('Grouping players by Overall', fontsize=30, fontweight='bold', y=1.05,)
plt.xlabel('Number of players', fontsize=25)
plt.ylabel('Players Age', fontsize=25)
sns.countplot(x="Overall", data=df, palette="hls");
plt.show()





def get_best_squad(formation):
    FIFA19_copy = df.copy()
    store = []
    
    # iterate through all positions in the input formation and get players with highest overall respective to the position
    for i in formation:
        store.append([
            i,
            FIFA19_copy.loc[[FIFA19_copy[FIFA19_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False),
            FIFA19_copy[FIFA19_copy['Position'] == i]['Overall'].max(),
            FIFA19_copy.loc[[FIFA19_copy[FIFA19_copy['Position'] == i]['Overall'].idxmax()]]['Age'].to_string(index = False),
            FIFA19_copy.loc[[FIFA19_copy[FIFA19_copy['Position'] == i]['Overall'].idxmax()]]['Club'].to_string(index = False),
            FIFA19_copy.loc[[FIFA19_copy[FIFA19_copy['Position'] == i]['Overall'].idxmax()]]['Value'].to_string(index = False),
            FIFA19_copy.loc[[FIFA19_copy[FIFA19_copy['Position'] == i]['Overall'].idxmax()]]['Wage'].to_string(index = False)
        ])
                      
        FIFA19_copy.drop(FIFA19_copy[FIFA19_copy['Position'] == i]['Overall'].idxmax(), 
                         inplace = True)
    

    return pd.DataFrame(np.array(store).reshape(11,7), 
                        columns = ['Position', 'Player', 'Overall', 'Age', 'Club', 'Value', 'Wage']).to_string(index = False)




print("Finding the Best Squad ")
squad_433 = ['GK', 'RB', 'CB', 'CB', 'LB', 'CDM', 'CM', 'CAM', 'RF', 'ST', 'LW']
print ('4-3-3')
print (get_best_squad(squad_433))

# 3-5-2
squad_352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'RF', 'LF']
print ('3-5-2')
print (get_best_squad(squad_352))


players = df[df.Overall > 85]
players = players.Club.value_counts()
plt.figure(figsize=(12,10))
sns.barplot(x=players.index,y=players.values)
plt.xticks(rotation=90)
plt.xlabel("Club")
plt.ylabel("No. of Players(Over 85)")




skills = ['Overall', 'Potential', 'Crossing',
   'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
   'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
   'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
   'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
   'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
   'Marking', 'StandingTackle', 'SlidingTackle']


print("Messi vs Ronaldo ")
messi = df.loc[df['Name'] == 'L. Messi']
messi = pd.DataFrame(messi, columns = skills)
ronaldo = df.loc[df['Name'] == 'Cristiano Ronaldo']
ronaldo = pd.DataFrame(ronaldo, columns = skills)

plt.figure(figsize=(14,8))
sns.pointplot(data=messi,color='blue',alpha=0.6)
sns.pointplot(data=ronaldo, color='red', alpha=0.6)
plt.text(5,55,'Messi',color='blue',fontsize = 25)
plt.text(5,50,'Ronaldo',color='red',fontsize = 25)
plt.xticks(rotation=90)
plt.xlabel('Skills', fontsize=20)
plt.ylabel('Skill value', fontsize=20)
plt.title('Messi vs Ronaldo', fontsize = 25)
plt.grid()
plt.show()


print("Age Distribution in Countries ")
countries_names = ('France', 'Brazil', 'Germany', 'Belgium', 'Spain', 'Netherlands', 'Argentina', 'Portugal', 'Chile', 'Colombia')
countries = df.loc[df['Nationality'].isin(countries_names) & df['Age']]
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
ax = sns.boxplot(x="Nationality", y="Age", data=countries)
ax.set_title(label='Age distribution in countries', fontsize=25)
plt.xlabel('Countries', fontsize=20)
plt.ylabel('Age', fontsize=20)
plt.grid()
plt.show()

