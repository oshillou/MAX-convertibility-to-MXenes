import argparse
import os
import pandas as pd
from ase.formula import Formula

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_calculated", type=str, required=True)
    parser.add_argument("--max_elemental", type=str, required=True)
    parser.add_argument("--max_synthesized", type=str, required=True)

    parser.add_argument("--output_continuous", type=str, default="continuous_data.csv")
    parser.add_argument("--output_categorical", type=str, default="categorical_data.csv")
    parser.add_argument("--output_targets", type=str, default="targets.csv")

    args = parser.parse_args()

    assert os.path.exists(args.max_calculated)
    assert os.path.exists(args.max_elemental)
    assert os.path.exists(args.max_synthesized)

    return args

def max2mx(max_name):
    max_counts = Formula(max_name).count()
    del max_counts[list(max_counts)[1]] # A element is always in position 1
    mx_name_unsorted = "".join([f"{atom}{count}" for atom, count in max_counts.items()])
    return Formula(mx_name_unsorted).format("metal")

def main():
    args = get_args()

    print("Loading the data")
    calculated_df = pd.read_csv(args.max_calculated)
    elemental_df = pd.read_csv(args.max_elemental)
    mxene_df = pd.read_excel(args.max_synthesized, sheet_name=0)

    print("Dropping elements")
    mxene_df.dropna(subset="MXene", inplace=True)
    mxene_df.drop(['Unnamed: 9','Unnamed: 12','Notes','status','Reference method'],axis=1, inplace=True)

    print("Building the PU class")
    parents = mxene_df["Parent material"].unique()
    banned_words = ['+','Mxene','topochemical','reaction', 'or', 'synthesis','MXene','direct']
    complete_parents = [word for parent in parents for word in filter(lambda x: x not in banned_words, parent.split())]
    # We must create the class. Any non-null integer will mean that the material is synthesizable
    # The number then refers to the unique category of MXene (several MAX phases can lead to the same MXene) 
    is_synthesizable = elemental_df["compound_formula"].isin(complete_parents)
    mxene_names = [max2mx(x) for x in elemental_df.loc[is_synthesizable, "compound_formula"]]
    unique_mxenes = list(set(mxene_names))
    mxene_group = [unique_mxenes.index(x)+1 for x in mxene_names]
    elemental_df["class"] = 0
    elemental_df.loc[is_synthesizable, "class"] = mxene_group

    print("Removing unnecessary columns")

    elemental_df.drop(['M_element', 'A_element', 'X_element'], axis=1, inplace=True)
    calculated_df.drop(["M","A","X","PU_label","year"], axis=1, inplace=True)

    print("Removing redundant columns")
    m_columns_to_drop = ["M_M_atomic_number", "M_M_molar_volume", "M_M_density", "M_M_average_cationic_radius"]
    a_columns_to_drop = ["A_A_atomic_number", "A_A_molar_volume", "A_A_density"]
    x_columns_to_drop = [x for x in elemental_df if f"X_"==x[:2]]
    x_columns_to_drop = list(filter(lambda x: x not in ["X_X_atomic_mass", "X_X_atomic_radius"], x_columns_to_drop))
    elemental_df.drop(m_columns_to_drop+a_columns_to_drop+x_columns_to_drop, axis=1, inplace=True)

    print("Setting compound as index")
    elemental_df = elemental_df.set_index('compound_formula',drop=True)
    calculated_df = calculated_df.set_index('prettyformula',drop=True)


    print("Merging dataframe")
    whole_data = elemental_df.merge(calculated_df,how='inner',
                                    left_index=True,right_index=True)
    
    print("Creating dummies for categorical variables")
    a_group = pd.get_dummies(whole_data['A_A_group'],prefix='a_g',dtype=float)
    m_group = pd.get_dummies(whole_data['M_M_group'],prefix='m_g',dtype=float)
    whole_data.drop(['A_A_group','M_M_group'],axis=1, inplace=True)
    whole_data = pd.concat([whole_data, a_group, m_group], axis=1)

    print("Splitting between ategorical/continuous/target variables and exporting")
    categorical_variables = a_group.columns.to_list()+m_group.columns.to_list()
    continuous_variables = [x for x in whole_data.columns if x not in categorical_variables+["class"]]

    whole_data[categorical_variables].to_csv(args.output_categorical)
    whole_data[continuous_variables].to_csv(args.output_continuous)
    whole_data["class"].to_csv(args.output_targets)

if __name__ == "__main__":
    main()
