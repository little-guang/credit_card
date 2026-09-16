-- 1. 性別對照表
CREATE TABLE SEX (
    SEX INT PRIMARY KEY,
    SEX_Name VARCHAR(20) NOT NULL
);

INSERT INTO SEX (SEX, SEX_Name)
VALUES
(1, '男性'),
(2, '女性');


-- 2. 教育程度對照表
CREATE TABLE Education (
    EDUCATION INT PRIMARY KEY,
    EDUCATION_Name VARCHAR(20) NOT NULL
);

INSERT INTO Education (EDUCATION, EDUCATION_Name)
VALUES
(1, '研究所'),
(2, '大學'),
(3, '高中'),
(4, '其他');


-- 3. 婚姻狀態對照表
CREATE TABLE Marital_Status (
    MARRIAGE INT PRIMARY KEY,
    MARRIAGE_Name VARCHAR(20) NOT NULL
);

INSERT INTO Marital_Status (MARRIAGE, MARRIAGE_Name)
VALUES
(1, '已婚'),
(2, '單身'),
(3, '其他');


-- 4. 還款狀態對照表
CREATE TABLE Payment_Status (
    PAY_Status INT PRIMARY KEY,
    PAY_Status_Name VARCHAR(30) NOT NULL
);

INSERT INTO Payment_Status (PAY_Status, PAY_Status_Name)
VALUES
(-1, '正常還款'),
(1, '延遲1個月'),
(2, '延遲2個月'),
(3, '延遲3個月'),
(4, '延遲4個月'),
(5, '延遲5個月'),
(6, '延遲6個月'),
(7, '延遲7個月'),
(8, '延遲8個月'),
(9, '延遲9個月以上');


-- 5. 違約結果對照表
CREATE TABLE Default_Status (
    default_payment_next_month INT PRIMARY KEY,
    Default_Status_Name VARCHAR(20) NOT NULL
);

INSERT INTO Default_Status (default_payment_next_month, Default_Status_Name)
VALUES
(0, '未違約'),
(1, '違約');