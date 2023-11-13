import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *
import nltk

# Portfolio returns and Sharp ratios

e_r1 = 0.05
e_r2 = 0.12
cor_r1_r2 = -0.6
st_dev_r1 = 0.05
st_dev_r2 = 0.10
rf = 0.02


def portfolio_analysis(r1, r2, cr12, sd1, sd2, rfr: float):
    def sharp_ratio(return_asset: float, risk_free_rate: float, st_dev_asset: float):
        result = (return_asset - risk_free_rate) / st_dev_asset
        return round(result, 6)

    def e_portfolio_return(weight: float, e_return1: float, e_return2: float):
        result: float = weight * e_return1 + (1 - weight) * e_return2
        return round(result, 6)

    def portfolio_variance(weight: float, st_dev_1: float, st_dev_2: float, correlation: float):
        result = (weight ** 2) * (st_dev_1 ** 2) + \
                 ((1 - weight) ** 2) * (st_dev_2 ** 2) + \
                 2 * correlation * weight * (1 - weight) * st_dev_1 * st_dev_2
        return round(result, 6)

    sharp_ratio1 = sharp_ratio(e_r1, rf, st_dev_r1)
    sharp_ratio2 = sharp_ratio(e_r2, rf, st_dev_r2)

    print(f"Sharp ration 1: {sharp_ratio1}")
    print(f"Sharp ration 2: {sharp_ratio2}")

    # weights for portfolios
    weights = [round(i / 100, 2) for i in range(0, 101, 10)]

    # st dev for portfolios
    st_dev_portfolios = [
        round(np.sqrt(portfolio_variance(weight=weight, st_dev_1=st_dev_r1, st_dev_2=st_dev_r2, correlation=cor_r1_r2)),
              4)
        for weight in weights]

    # returns for portfolios
    returns_portfolios = [round(e_portfolio_return(weight=weight, e_return1=e_r1, e_return2=e_r2), 4)
                          for weight in weights]
    sharp_ratio_portfolios = [round(sharp_ratio(return_asset=returns_portfolios[i],
                                          st_dev_asset=st_dev_portfolios[i],
                                          risk_free_rate=rf), 4)
                              for i in range(0, len(weights))]

    # Finding optimised portfolio
    optimised_portfolio_index = sharp_ratio_portfolios.index(max(sharp_ratio_portfolios))
    optimised_portfolio_return = returns_portfolios[optimised_portfolio_index]
    optimised_portfolio_st_dev = st_dev_portfolios[optimised_portfolio_index]
    optimised_weight = weights[optimised_portfolio_index]
    optimised_sharp_ration = sharp_ratio_portfolios[optimised_portfolio_index]

    print(f'Weights list: {weights}')
    print(f'Returns list: {returns_portfolios}')
    print(f'St dev list: {st_dev_portfolios}')
    print(f'Sharp ratio list: {sharp_ratio_portfolios}')
    print(f'Optimised weight: {optimised_weight}')
    print(f'Optimised portfolio return: {optimised_portfolio_return}')
    print(f'Optimised portfolio st dev: {optimised_portfolio_st_dev}')
    print(f'Optimised portfolio sharp ratio: {optimised_sharp_ration}')


portfolio_analysis(r1=e_r1, r2=e_r2, cr12=cor_r1_r2, sd1=st_dev_r1, sd2=st_dev_r2, rfr=rf)

# Plotting data for growth

growth_data = pd.read_csv('ec_growth.csv')
credit_rights = pd.read_excel('jfe_2007__dataset_oct08.xls', sheet_name='data')
print(growth_data)
growth_after_1980 = growth_data[growth_data['year'] > 1980]
print(growth_after_1980)

information_sharing_pc_gdp = credit_rights[['pc_gdp', 'info']].groupby('info').mean()
print(information_sharing_pc_gdp)

country1 = "China"
country2 = "India"
plt.ylabel("Economic growth", color="black", fontsize=14)
plt.plot(growth_after_1980.year[growth_after_1980['countryname'] == country1],
         growth_after_1980.ec_growth[growth_after_1980['countryname'] == country1])
plt.plot(growth_after_1980.year[growth_after_1980['countryname'] == country2],
         growth_after_1980.ec_growth[growth_after_1980['countryname'] == country2])
plt.show()

# Plotting data for trillema

path = "trilemma_indexes_update2020.xlsx"
data = pd.read_excel(path)

data_1980_2020 = data[data['year'] > 1980]

country1 = "China"
fig,ax=plt.subplots()
plt.title("Policy Trilemma choices in " + country1)
ax.set_ylabel("Index",color="black",fontsize=14)
ax.plot(data.year[data['Country Name']==country1], data['Exchange Rate Stability Index'][data['Country Name'] == country1], color="red", label="Exchange Rate Stability Index")
ax.plot(data.year[data['Country Name']==country1], data['Monetary Independence Index'][data['Country Name'] == country1], color="blue", label="Monetary Independence Index")
ax.plot(data.year[data['Country Name']==country1], data['Financial Openness Index'][data['Country Name'] == country1], color="green", label="Financial Openness Index")
plt.legend(bbox_to_anchor=(0.8, -0.1))
plt.show()

# IS - LM calculations

Ex, r, Y = symbols('Ex r Y')

I = '300 - 3000*r'
G = '300'
T = '300'
C = f"2000 + 0.6*(Y - {T})"
NX = '400 - 200*Ex'
Ms = '500'
Md = '0.2*Y - 1000*r'
Ex_value = 2

IS = f"{C} + {I} + {G} + {NX} - Y"
LM = f"{Md} - {Ms}"
print(f"IS curve calculations: {IS} = 0")
print(f"LM curve calculations: {LM} = 0")

IS_solved = solve(IS, Y)[0]
LM_solved = solve(LM, Y)[0]
print(f"IS curve: Y = {IS_solved}")
print(f"LM curve: Y = {LM_solved}")

equality = f"{IS_solved} - ({LM_solved})"
equality_solved = solve(equality, r)[0]
r_optimal = float(equality_solved.subs(Ex, Ex_value))
y_optimal = float(LM_solved.subs(r, r_optimal))
print(f"Optimal r = {r_optimal}")
print(f"Optimal y = {y_optimal}")

# Covered interest parity

s = 1.6  # spot exchange rate
ind = 0.08  # domestic interest rate
domestic_currency = "USD"
f = 1.63  # forward exchange rate
inf = 0.12  # foreign interest rate
foreign_currency = "Pounds"
amount_money = 1000  # in thousands

if s == f*(1+inf)/(1+ind):
    print("Exchange rate parity holds")
    arbitrage = False
else:
    print("Exchange rate parity doesn't hold, so there is an arbitrage")
    arbitrage = True

if arbitrage:
    spot_side = s
    forward_side = f*(1+inf)/(1+ind)
    if forward_side < spot_side:
        to_be_paid = amount_money*(1+inf)
        conversion_to_domestic = round(amount_money*s, 3)
        after_domestic_investment = round(conversion_to_domestic*(1+ind), 3)
        conversion_back_to_foreign = round(after_domestic_investment/f, 3)
        profit = round(conversion_back_to_foreign - to_be_paid, 3)
        print(f'We need to invest in domestic currency\n'
              f' 1. Borrow {foreign_currency}: to be paid {to_be_paid} {foreign_currency},\n'
              f' 2. Convert {foreign_currency} to {domestic_currency}: {conversion_to_domestic} {domestic_currency},\n'
              f' 3. Invest at domestic interest rate: {after_domestic_investment} {domestic_currency},\n'
              f' 4. Convert back to foreign at forward rate, {conversion_back_to_foreign} {foreign_currency},\n'
              f' 5. Calculate the difference between money to pay and money after conversion: {profit} {foreign_currency}')
    elif forward_side > spot_side:
        to_be_paid = amount_money * (1 + ind)
        conversion_to_foreign = round(amount_money / s, 3)
        after_foreign_investment = round(conversion_to_foreign * (1 + inf), 3)
        conversion_back_to_domestic = round(after_foreign_investment * f, 3)
        profit = round(conversion_back_to_domestic - to_be_paid, 3)
        print(f'We need to invest in foreign currency\n'
              f' 1. Borrow {domestic_currency}: to be paid {to_be_paid} {domestic_currency},\n'
              f' 2. Convert {domestic_currency} to {foreign_currency}: {conversion_to_foreign} {foreign_currency},\n'
              f' 3. Invest at foreign interest rate: {after_foreign_investment} {foreign_currency},\n'
              f' 4. Convert back to domestic at forward rate, {conversion_back_to_domestic} {domestic_currency},\n'
              f' 5. Calculate the difference between money to pay and money after conversion: {profit} {domestic_currency}')

# Calculating forward exchange rate

s = 1.6  # spot exchange rate
ind = 0.08  # domestic interest rate
inf = 0.12  # foreign interest rate

f = s*(1+ind)/(1+inf)
print(f"Foreign exchange rate = {f}")


# Calculating spot exchange rate

ind = 0.08  # domestic interest rate
f = 1.63  # forward exchange rate
inf = 0.12  # foreign interest rate

s = f*(1+inf)/(1+ind)
print(f"Spot exchange rate = {s}")

# Calculating domestic interest rate

s = 1.6  # spot exchange rate
f = 1.63  # forward exchange rate
inf = 0.12  # foreign interest rate

ind = f*(1+inf)/s - 1
print(ind)

# Calculating foreign interest rate

s = 1.6  # spot exchange rate
f = 1.63  # forward exchange rate
ind = 0.08  # domestic interest rate

inf = s*(1+ind)/f - 1
print(inf)

# Calculating investor return taking into account currency risk

s = 0.0245  # foreign/domestic spot rate
domestic_currency = "Pounds"
f = 0.0222  # foreign/domestic forward rate
inf = 0.07  # foreign rate on foreign investment
foreign_currency = "Thai Bath"

change_in_foreign_currency = round(f/s - 1, 4)
print(f"Change in the foreign currency: {change_in_foreign_currency}")
total_investor_return = round((1+inf)*(1+change_in_foreign_currency)-1, 4)
print(f"Total investor return: {total_investor_return}")

# CDS spread calculations

rf = 0.04  # risk-free rate
principal = 1  # principal in 1 mln
c = 0.3  # recovery rate
t = 3  # number of years
default_rate = 0.01  # default rate

time_list = []
default_probability_list = []
survival_probability_list = []
expected_payments_list = []
discount_factor_list = []
pv_payments_list = []
expected_payoffs_list = []
pv_payoffs_list = []

for i in range(1, t+1):
    time_list.append(i)
    default_prob = default_rate*((1-default_rate)**(i-1))
    default_probability_list.append(default_prob)

    survival_probability = (1-default_rate)**i
    survival_probability_list.append(survival_probability)

    discount_factor = 1/((1+rf)**i)
    discount_factor_list.append(discount_factor)

    expected_payment = survival_probability
    expected_payments_list.append(expected_payment)

    pv_payment = expected_payment*discount_factor
    pv_payments_list.append(pv_payment)

    expected_payoff = (1-c)*default_prob
    expected_payoffs_list.append(expected_payoff)

    pv_payoff = expected_payoff*discount_factor
    pv_payoffs_list.append(pv_payoff)

answer_dict = {
    'Time': time_list,
    'Pauments': expected_payments_list,
    'Payoffs': expected_payoffs_list,
    'DF': discount_factor_list,
    'PV Paym': pv_payments_list,
    'PV Payoff': pv_payoffs_list
}
answer_table = pd.DataFrame.from_dict(answer_dict)

total_payments = round(sum(pv_payments_list), 4)
total_payoffs = round(sum(pv_payoffs_list), 4)
spread = round(total_payoffs/total_payments, 4)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 5,
                       ):
    print(answer_table, '\n')

print(f'Total payments: {total_payments}, Total payoffs: {total_payoffs}, CDS spread: {spread} ')

# Re-levering beta (bunlev)

d = 7.4  # debt
e = 4  # equity
rmp = 0.053  # market risk premium
rcp = 0.0264  # country risk premium
rf = 0.0175  # risk-free rate
t = 0.22  # tax
rd = 0.0685  # cost of debt
blev = 0.6  # levered or equity beta of an industry
bunlev = blev/(1+(1-t)*d/e)
print(bunlev)

# Re-levering beta (blev)

d = 7.4  # debt
e = 4  # equity
rmp = 0.053  # market risk premium
rcp = 0.0264  # country risk premium
rf = 0.0175  # risk-free rate
t = 0.22  # tax
rd = 0.0685  # cost of debt
bunlev = 0.6  # levered or asset beta of an industry
blev = bunlev*(1+(1-t)*(d/e))
print(blev)

# optimal hedge ration

unhedged_asset_variance = 0.3
future_variance = 0.5
correlation = 0.7

optimal_portfolio_hedge = correlation*np.sqrt(unhedged_asset_variance)/np.sqrt(future_variance)
print(optimal_portfolio_hedge)

# Text analysis

nltk.download('punkt')

article = 'Timeline: Thailand s turbulent politics since 2014 military coup By Reuters Staff BANGKOK (Reuters)'
number_senteces = article.count('.')
tokens = nltk.word_tokenize(article)
total_words = len(tokens)
print(total_words)
wordcount = {}
wordfreq = {}

for token in tokens:
    if token not in wordfreq.keys():
        wordfreq[token] = 1
    else:
        wordfreq[token] += 1

for word, freq in wordfreq.items():
    wordcount[word] = freq
    wordfreq[word] = freq / total_words

print(wordcount)
print(wordfreq)

# Microfinance - lending
investment = 1
money_return_safe = 2.5
money_return_risky = 3.125
probability_safe = 1
probability_risky = 0.8
good_project_share = 0.3
reserve_utility = 1.4
symmetric = False
joint_liability = True
joint_interest = 1
default_payment = 1.25


def investment_decision(utility, reservational_utility):
    if utility > reservational_utility:
        decision = 'Investor would finance'
    elif utility == reservational_utility:
        decision = 'Investor is indifferent between financing and not financing'
    else:
        decision = 'Investor would not finance'
    return decision


if not symmetric:
    as_prob = good_project_share*probability_safe+(1-good_project_share)*probability_risky
    bank_interest = 1/as_prob
    utility_safe = probability_safe*(money_return_safe-bank_interest)
    utility_risky = probability_risky*(money_return_risky-bank_interest)

    decision_safe = investment_decision(utility=utility_safe, reservational_utility=reserve_utility)
    decision_risky = investment_decision(utility=utility_risky, reservational_utility=reserve_utility)

    print(f'Asymmetric information:\n'
          f'm = {round(as_prob, 2)}\n'
          f'bank interest = {round(bank_interest, 2)}\n'
          f'Reservation utility = {round(reserve_utility, 2)}\n'
          f'Utility of the safe investor = {round(utility_safe, 2)}\n'
          f'Decision of the safe investor - {decision_safe}\n'
          f'Utility of the risky investor = {round(utility_risky, 2)}\n'
          f'Decision of the risky investor - {decision_risky}\n')

else:
    bank_interest_safe = 1 / probability_safe
    bank_interest_risky = 1/ probability_risky
    utility_safe = probability_safe * (money_return_safe - bank_interest_safe)
    utility_risky = probability_risky * (money_return_risky - bank_interest_risky)
    decision_safe = investment_decision(utility=utility_safe, reservational_utility=reserve_utility)
    decision_risky = investment_decision(utility=utility_risky, reservational_utility=reserve_utility)

    print(f'Symmetric information:\n'
          f'bank interest safe = {round(bank_interest_safe, 2)}\n'
          f'bank interest risky = {round(bank_interest_risky, 2)}\n'
          f'Reservation utility = {round(reserve_utility, 2)}\n'
          f'Utility of the safe investor = {round(utility_safe, 2)}\n'
          f'Decision of the safe investor - {decision_safe}\n'
          f'Utility of the risky investor = {round(utility_risky, 2)}\n'
          f'Decision of the risky investor - {decision_risky}\n')

if joint_liability:
    uss = probability_safe*money_return_safe - (probability_safe*joint_interest+probability_safe*(1 - probability_safe)*default_payment)
    usr = probability_safe*money_return_safe - (probability_safe*joint_interest+probability_safe*(1 - probability_risky)*default_payment)
    urr = probability_risky*money_return_risky - (probability_risky*joint_interest+probability_risky*(1 - probability_risky)*default_payment)
    urs = probability_risky*money_return_risky - (probability_risky*joint_interest+probability_risky*(1 - probability_safe)*default_payment)
    dif_uss_usr = uss - usr
    dif_urr_urs = urr - urs

    decision_uss = investment_decision(utility=uss, reservational_utility=reserve_utility)
    decision_urr = investment_decision(utility=urr, reservational_utility=reserve_utility)

    print(f'Joint liability:\n'
          f'bank joint interest = {round(joint_interest, 2)}\n'
          f'bank default payment = {round(default_payment, 2)}\n'
          f'Reservation utility = {round(reserve_utility, 2)}\n'
          f'Utility of the safe investor with safe investor = {round(uss, 2)}\n'
          f'Decision of the safe investor - {decision_uss}\n'
          f'Utility of the risky investor with risky investor = {round(urr, 2)}\n'
          f'Decision of the risky investor - {decision_urr}\n'
          f'usr = {round(usr, 2)}\n'
          f'urs = {round(urs, 2)}\n'
          f'Difference between uss and usr = {round(dif_uss_usr, 2)}\n'
          f'Difference between urr and urs = {round(dif_urr_urs, 2)}\n')

# Bank run

bank_reserves = 10
trader1_reserves = 10
trader2_reserves = 6
traders_cost = 1
devaluation_percentage = 0.50

if trader1_reserves+trader2_reserves < bank_reserves:
    print('Traders will not cause the bank run due to insufficient funds')
else:
    hold_hold_1 = 0
    hold_hold_2 = 0
    if (trader1_reserves >= bank_reserves) or (trader2_reserves >= bank_reserves):
        hold_sell_1 = 0
        hold_sell_2 = bank_reserves*devaluation_percentage - traders_cost
        sell_hold_1 = bank_reserves*devaluation_percentage - traders_cost
        sell_hold_2 = 0
        sell_sell_1 = (bank_reserves/2)*devaluation_percentage - traders_cost
        sell_sell_2 = (bank_reserves/2)*devaluation_percentage - traders_cost
        print(f'hold_sell_1 = 0\n'
              f'hold_sell_2 = bank_reserves*devaluation_percentage - traders_cost = {bank_reserves}*{devaluation_percentage} - {traders_cost} = {hold_sell_1}\n'
              f'sell_hold_1 = bank_reserves*devaluation_percentage - traders_cost = {bank_reserves}*{devaluation_percentage} - {traders_cost} = {sell_hold_1}\n'
              f'sell_hold_2 = 0\n'
              f'sell_sell_1 = (bank_reserves/2)*devaluation_percentage - traders_cost = ({bank_reserves}/2)*{devaluation_percentage} - {traders_cost} = {sell_sell_1}\n'
              f'sell_sell_2 = (bank_reserves/2)*devaluation_percentage - traders_cost = ({bank_reserves}/2)*{devaluation_percentage} - {traders_cost} = {sell_sell_2}\n')
    else:
        hold_sell_1 = 0
        hold_sell_2 = - traders_cost
        sell_hold_1 = - traders_cost
        sell_hold_2 = 0
        sell_sell_1 = (bank_reserves/2)*devaluation_percentage - traders_cost
        sell_sell_2 = (bank_reserves/2)*devaluation_percentage - traders_cost
        print(f'hold_sell_1 = 0\n'
              f'hold_sell_2 = - traders_cost = {-traders_cost}\n'
              f'sell_hold_1 = - traders_cost = {-traders_cost}\n'
              f'sell_hold_2 = 0\n'
              f'sell_sell_1 = (bank_reserves/2)*devaluation_percentage - traders_cost = ({bank_reserves}/2)*{devaluation_percentage} - {traders_cost} = {sell_sell_1}\n'
              f'sell_sell_2 = (bank_reserves/2)*devaluation_percentage - traders_cost = ({bank_reserves}/2)*{devaluation_percentage} - {traders_cost} = {sell_sell_2}\n')

    print(f'      Hold     Sell\n'
          f' Hold {hold_hold_1} , {hold_hold_2} | {hold_sell_1} , {hold_sell_2}\n'
          f' Sell {sell_hold_1} , {sell_hold_2} | {sell_sell_1} , {sell_sell_2}\n')

# Diamond-Dybvig (1983) model

number_consumers = 100
consumers_investment = 1
impatient_consumers = 0.25
rate_1_period = 1.28
