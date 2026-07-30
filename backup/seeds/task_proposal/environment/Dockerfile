FROM ubuntu:24.04

RUN apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# No header row. Four valid lines summing to 61.75, three malformed:
#   line 3 has a non-numeric amount, line 5 an empty currency, line 7 a missing field.
RUN printf '%s\n' \
    '1,10.50,USD' \
    '2,20.25,EUR' \
    '3,not_a_number,USD' \
    '4,5.00,GBP' \
    '5,26.00,' \
    '6,26.00,JPY' \
    '7,15.00' \
    > /app/records.csv
