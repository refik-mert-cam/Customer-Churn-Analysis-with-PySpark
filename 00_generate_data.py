import argparse
import os
import random
import csv

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + pow(2.718281828, -x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of rows to generate")
    parser.add_argument("--out", type=str, default="data/churn.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    header = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "InternetService", "Contract",
        "MonthlyCharges", "TotalCharges", "Churn"
    ]

    genders = ["Female", "Male"]
    yesno = ["Yes", "No"]
    internet = ["DSL", "Fiber", "None"]
    contract = ["Month-to-month", "One year", "Two year"]

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(args.n):
            customer_id = f"C{100000+i}"
            gender = random.choice(genders)
            senior = 1 if random.random() < 0.18 else 0
            partner = random.choice(yesno)
            dependents = "Yes" if (partner == "Yes" and random.random() < 0.5) else "No"

            # tenure: skewed to lower values
            tenure = int(min(72, max(0, random.expovariate(1/18))))
            phone = "Yes" if random.random() < 0.9 else "No"
            net = random.choices(internet, weights=[0.35, 0.5, 0.15], k=1)[0]
            ctr = random.choices(contract, weights=[0.55, 0.25, 0.20], k=1)[0]

            # monthly charges: influenced by internet type
            base = 20.0
            if net == "DSL":
                base += 25.0
            elif net == "Fiber":
                base += 45.0
            else:
                base += 0.0

            if phone == "Yes":
                base += 8.0

            # noise + small senior effect
            monthly = max(15.0, base + random.gauss(0, 8) + (5.0 if senior else 0.0))
            total = round(monthly * max(tenure, 1) + random.gauss(0, 25), 2)

            # churn probability model (synthetic but realistic-ish)
            # Higher churn for month-to-month, fiber, low tenure; lower churn for long contracts and high tenure.
            z = 0.0
            z += 1.1 if ctr == "Month-to-month" else (-0.3 if ctr == "One year" else -0.8)
            z += 0.6 if net == "Fiber" else (0.2 if net == "DSL" else -0.2)
            z += 0.4 if senior else 0.0
            z += -0.03 * tenure
            z += 0.01 * (monthly - 50.0)
            z += random.gauss(0, 0.35)

            p_churn = sigmoid(z)
            churn = "Yes" if random.random() < p_churn else "No"

            writer.writerow([
                customer_id, gender, senior, partner, dependents,
                tenure, phone, net, ctr,
                round(monthly, 2), total, churn
            ])

    print(f"Wrote {args.n} rows to {args.out}")

if __name__ == "__main__":
    main()
