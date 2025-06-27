create table if not exists buildings
(
    name varchar(100) not null
        primary key
);

create table if not exists predictions
(
    building_name varchar(100) not null
        references buildings,
    target_date   date         not null,
    h0            real,
    h1            real,
    h2            real,
    h3            real,
    h4            real,
    h5            real,
    h6            real,
    h7            real,
    h8            real,
    h9            real,
    h10           real,
    h11           real,
    h12           real,
    h13           real,
    h14           real,
    h15           real,
    h16           real,
    h17           real,
    h18           real,
    h19           real,
    h20           real,
    h21           real,
    h22           real,
    h23           real,
    mse           real,
    mae           real,
    r2            real,
    smape         real,
    primary key (building_name, target_date)
);

create table if not exists realconsumption
(
    building_name text not null
        references buildings,
    target_date   date not null,
    h0            real,
    h1            real,
    h2            real,
    h3            real,
    h4            real,
    h5            real,
    h6            real,
    h7            real,
    h8            real,
    h9            real,
    h10           real,
    h11           real,
    h12           real,
    h13           real,
    h14           real,
    h15           real,
    h16           real,
    h17           real,
    h18           real,
    h19           real,
    h20           real,
    h21           real,
    h22           real,
    h23           real,
    primary key (building_name, target_date)
);

create table if not exists baselines
(
    building_name varchar(100) not null
        references buildings,
    target_date   date         not null,
    h0            real,
    h1            real,
    h2            real,
    h3            real,
    h4            real,
    h5            real,
    h6            real,
    h7            real,
    h8            real,
    h9            real,
    h10           real,
    h11           real,
    h12           real,
    h13           real,
    h14           real,
    h15           real,
    h16           real,
    h17           real,
    h18           real,
    h19           real,
    h20           real,
    h21           real,
    h22           real,
    h23           real,
    primary key (building_name, target_date)
);

create table if not exists airtemperature
(
    target_date date not null
        constraint weather_pkey
            primary key,
    h0          real,
    h1          real,
    h2          real,
    h3          real,
    h4          real,
    h5          real,
    h6          real,
    h7          real,
    h8          real,
    h9          real,
    h10         real,
    h11         real,
    h12         real,
    h13         real,
    h14         real,
    h15         real,
    h16         real,
    h17         real,
    h18         real,
    h19         real,
    h20         real,
    h21         real,
    h22         real,
    h23         real
);

create table if not exists dewtemperature
(
    target_date date not null
        primary key,
    h0          real,
    h1          real,
    h2          real,
    h3          real,
    h4          real,
    h5          real,
    h6          real,
    h7          real,
    h8          real,
    h9          real,
    h10         real,
    h11         real,
    h12         real,
    h13         real,
    h14         real,
    h15         real,
    h16         real,
    h17         real,
    h18         real,
    h19         real,
    h20         real,
    h21         real,
    h22         real,
    h23         real
);

create table if not exists gaoptimzation
(
    building_name text not null
        references buildings,
    target_date   date not null,
    h0            real,
    h1            real,
    h2            real,
    h3            real,
    h4            real,
    h5            real,
    h6            real,
    h7            real,
    h8            real,
    h9            real,
    h10           real,
    h11           real,
    h12           real,
    h13           real,
    h14           real,
    h15           real,
    h16           real,
    h17           real,
    h18           real,
    h19           real,
    h20           real,
    h21           real,
    h22           real,
    h23           real,
    primary key (building_name, target_date)
);

create table if not exists acooptimzation
(
    building_name text not null
        references buildings,
    target_date   date not null,
    h0            real,
    h1            real,
    h2            real,
    h3            real,
    h4            real,
    h5            real,
    h6            real,
    h7            real,
    h8            real,
    h9            real,
    h10           real,
    h11           real,
    h12           real,
    h13           real,
    h14           real,
    h15           real,
    h16           real,
    h17           real,
    h18           real,
    h19           real,
    h20           real,
    h21           real,
    h22           real,
    h23           real,
    primary key (building_name, target_date)
);

create table if not exists psooptimzation
(
    building_name text not null
        references buildings,
    target_date   date not null,
    h0            real,
    h1            real,
    h2            real,
    h3            real,
    h4            real,
    h5            real,
    h6            real,
    h7            real,
    h8            real,
    h9            real,
    h10           real,
    h11           real,
    h12           real,
    h13           real,
    h14           real,
    h15           real,
    h16           real,
    h17           real,
    h18           real,
    h19           real,
    h20           real,
    h21           real,
    h22           real,
    h23           real,
    primary key (building_name, target_date)
);

create table if not exists windspeed
(
    target_date date not null
        primary key,
    h0          real,
    h1          real,
    h2          real,
    h3          real,
    h4          real,
    h5          real,
    h6          real,
    h7          real,
    h8          real,
    h9          real,
    h10         real,
    h11         real,
    h12         real,
    h13         real,
    h14         real,
    h15         real,
    h16         real,
    h17         real,
    h18         real,
    h19         real,
    h20         real,
    h21         real,
    h22         real,
    h23         real
);

create table if not exists winddirection
(
    target_date date not null
        primary key,
    h0          real,
    h1          real,
    h2          real,
    h3          real,
    h4          real,
    h5          real,
    h6          real,
    h7          real,
    h8          real,
    h9          real,
    h10         real,
    h11         real,
    h12         real,
    h13         real,
    h14         real,
    h15         real,
    h16         real,
    h17         real,
    h18         real,
    h19         real,
    h20         real,
    h21         real,
    h22         real,
    h23         real
);

create table if not exists precipdepth1hr
(
    target_date date not null
        primary key,
    h0          real,
    h1          real,
    h2          real,
    h3          real,
    h4          real,
    h5          real,
    h6          real,
    h7          real,
    h8          real,
    h9          real,
    h10         real,
    h11         real,
    h12         real,
    h13         real,
    h14         real,
    h15         real,
    h16         real,
    h17         real,
    h18         real,
    h19         real,
    h20         real,
    h21         real,
    h22         real,
    h23         real
);

