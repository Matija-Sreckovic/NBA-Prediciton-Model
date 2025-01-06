Extract Box-Scores zip file into the same (data) folder. Run either main.py or notebook.ipynb.

Recommended: the first entry should be 2018 (resp. 2019). The second entry should be 2022 (resp. 2023).

To see the predictions of each game, open "test_data_" + 2018/2019 + "_" + 2022/2023 + ".csv" after running the program and look at the last column (will_home_win?).

To see the players' ratings, open "players_test_data_" + 2018/2019 + "_" + 2022/2023 + ".csv"


## Explanation of the rating system

### GameScore, Off/DefRtg

I'll try to explain how the rating system works here. It uses two catch-all advanced stats of a box score: a player's **GameScore** (**GmSc**) and the difference between his **OffRtg** and **DefRtg**. The formula for **GmSc** is:

$$ GmSc = PTS + 0.4 \times FG - 0.7 \times FGA - 0.4\times(FTA - FT) + 0.7 \times ORB + 0.3 \times DRB + STL + 0.7 \times AST + 0.7 \times BLK - 0.4 \times PF - TOV.$$

On the other hand, **OffRtg** and **DefRtg** for a *team* are the number of points a team scores/allows per 100 possessions. For a *player*, the formula is much more complicated (see https://www.basketball-reference.com/about/ratings.html), but according to its creator, Dean Oliver, the author of ["Basketball on Paper"](https://www.amazon.com/Basketball-Paper-Rules-Performance-Analysis/dp/1574886886):

**OffRtg:** "Individual offensive rating is the number of points produced by a player per hundred total individual possessions. In other words, 'How many points is a player likely to generate when he tries?'"

**DefRtg:** "The core of the Defensive Rating calculation is the concept of the individual Defensive Stop. Stops take into account the instances of a player ending an opposing possession that are tracked in the boxscore (blocks, steals, and defensive rebounds), in addition to an estimate for the number of forced turnovers and forced misses by the player which aren't captured by steals and blocks."

### The Rating System Itself - 1 Game

A **score** is assigned to each player after each game he played in.

We take in each player's GmSc and difference between OffRtg and DefRtg (henceforth "Rtg"). We compare them to all other scores of players in all of the last 5 years - for example, if each is 99th percentile, the player gets a coefficient of 0.99 for GmSc and Rtg.

To get the player's *unweighted rating*, we set $\textup{rating}_{\textup{unweighted}} = 0.2 \times \textup{coeff}_{\textup{GmSc}} + 0.8 \times \textup{coeff}_{\textup{Rtg}}.$ The value $0.2$ was tuned.

To get the player's *weighted rating* we multiply the unweighted rating by a **usage rate coefficient** (we get the player's USG% and assign a coefficient by comparing it to all other games in the last 5 years, similarly to how we obtain the Gmsc and Rtg coefficients) and **minutes_coefficient** (if a player played at least 35 minutes, the coefficient is 1). Precisely, $\textup{coeff}_{\textup{min}} = 6.85 \times \frac{\textup{player's minutes played}}{\textup{total team minutes}}.$ In total, $$\textup{rating}_{\textup{weighted}} = \textup{coeff}_{\textup{USG%}} \times \textup{coeff}_{\textup{min}} \times (0.2 \times \textup{coeff}_{\textup{GmSc}} + 0.8 \times \textup{coeff}_{\textup{Rtg}}).$$

### Players' Long-Term Ratings and a Team's Rating before a Game

The score that we compute for a player after each game is added to his rating sum *this season* and his *5-year rating sum*. Both are divided by the number of games played to get the *average rating this season* and *average 5-year rating*. There are also *last season's average ratings* and *average ratings from 2 seasons ago*.

To compute a player's rating before a game, we use a weighted average of the 4 parameters above (this season's avg rating so far, last season's avg rating, avg rating from 2 seasons ago, 5-year avg rating). The weights depend on how many games the player has played this season and the previous 2 seasons. For example, if a player has played 15 games this year, 55 games last year and 0 games 2 years ago, the formula would be:

$$\textup{player game rating} = 0.5 \times \textup{rating this season} + 0.3 \times \textup{rating last season} + 0 \times \textup{rating 2 seasons ago} + 0.2 \times \textup{rating in the last 5 years}.$$

The weights were selected a bit arbitrarily, but tuning did not improve the model's performance.

At the end of this process, we get a vector of ratings for about 10-15 players; we take all **uninjured** players for the game! Call this vector $v_{\textup{ratings}}.$ The team's score is the dot product of this vector with the **vector of players' usage estimates**.

The players' usage estimates are numbers that take into account each player's average usage rate and minutes played. For example, if a player's average usage rate is 20% this season, 30% last season, and 27% in the last 5 years, and the player played 80 games both this season and the last, then

$$ \textup{player game usage coeff} = 0.6 \times \textup{this year's USG\%} + 0.2 \times \textup{last year's USG\%} + 0.2 \times \textup{5 year USG\%}. $$

There is also a minutes played usage coefficient, computed as follows: we take the player's average minutes coefficient computed earlier for both this season and last season (so if a player played at least 35 minutes in each game he played, both are $1$), compute a weight $\textup{minutes weight} =  \min(1, \textup{games played this season}  \times 0.1)$, and compute:

$$\textup{player game minutes coeff} = (\textup{minutes weight}) \times (\textup{avg minutes coeff this season}) + (1 - \textup{minutes weight}) \times (\textup{avg minutes coeff last season}).$$

Then we get each player's overall weight for the game by multiplying these:

$$\textup{player game weight} = \textup{player games usage coeff} \times \textup{player game minutes coeff}.$$

We take these game weights together into a vector $v_{\textup{weights}}$, and finally,

$$\textup{team score} = \langle v_{\textup{ratings}}, v_{\textup{weights}} \rangle.$$

Usually, the best players' ratings are around 40, with quick fall-offs towards the 20's. Negative ratings are not uncommon! The prediction is that the team with the greater score wins. There should also be a home-court advantage factor - the one that worked best for the 2023-24 season is +3 for the home team.
