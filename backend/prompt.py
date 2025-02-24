import pandas as pd

df = pd.read_csv('attack_prompts.csv')


melted_df = df.melt(
    id_vars=['Category'], 
    value_vars=['Prompt 1', 'Prompt 2', 'Prompt 3'],
    var_name='Prompt Number',
    value_name='Prompt'
)


melted_df['Prompt'] = melted_df['Prompt'].str.replace(r'^\*?\s*', '', regex=True).str.strip()


final_df = melted_df[['Category', 'Prompt']]
final_df.to_csv('cleaned_attack_prompts.csv', index=False)

print("Transformation complete! File saved as cleaned_attack_prompts.csv")