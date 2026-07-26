The file `/app/records.csv` contains one transaction per line in the form
`id,amount,currency`. There is no header row. Some lines are malformed: a missing
field, a non-numeric amount, or an empty currency.

Write `/app/summary.json` containing exactly two keys:

- `"total"`: the sum of `amount` over every **valid** line, rounded to 2 decimals.
- `"skipped"`: the number of malformed lines that were skipped.

Do not modify `records.csv`.
