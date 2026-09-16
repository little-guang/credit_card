/*建立資料庫credit_risk*/
CREATE DATABASE credit_risk;

/*使用資料庫credit_risk*/
USE credit_risk;

/*建立資料表credit_card_raw，設定欄位屬性*/
CREATE TABLE credit_card_raw (
    ID INT NOT NULL,
    LIMIT_BAL DECIMAL(12,2),
    SEX INT,
    EDUCATION INT,
    MARRIAGE INT,
    AGE INT,

    PAY_0 INT,
    PAY_2 INT,
    PAY_3 INT,
    PAY_4 INT,
    PAY_5 INT,
    PAY_6 INT,

    BILL_AMT1 DECIMAL(12,2),
    BILL_AMT2 DECIMAL(12,2),
    BILL_AMT3 DECIMAL(12,2),
    BILL_AMT4 DECIMAL(12,2),
    BILL_AMT5 DECIMAL(12,2),
    BILL_AMT6 DECIMAL(12,2),

    PAY_AMT1 DECIMAL(12,2),
    PAY_AMT2 DECIMAL(12,2),
    PAY_AMT3 DECIMAL(12,2),
    PAY_AMT4 DECIMAL(12,2),
    PAY_AMT5 DECIMAL(12,2),
    PAY_AMT6 DECIMAL(12,2),

    default_payment_next_month INT,

    PRIMARY KEY (ID)
);

/*載入資料來源credit_card.csv至資料表credit_card_raw*/
LOAD DATA LOCAL INFILE 'C:/proj/credit_card/credit_card.csv'
INTO TABLE credit_card_raw
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

/*計算資料表credit_card_raw中列數合*/
SELECT COUNT(*) AS total_rows
FROM credit_card_raw;

/*計算資料表credit_card_raw中的ID*/
SELECT COUNT(DISTINCT ID) AS unique_id
FROM credit_card_raw;

/*檢視資料表credit_card_raw中的前十筆資料*/
SELECT *
FROM credit_card_raw
LIMIT 10;

/*檢視資料表credit_card_raw中default_payment_next_month的總合*/
/*並依照default_payment_next_month排序*/
SELECT
    default_payment_next_month,
    COUNT(*) AS count
FROM credit_card_raw
GROUP BY default_payment_next_month;