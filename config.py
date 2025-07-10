DB_URI = 'postgresql://postgres:postgres@localhost:5432/licenta'

swagger_template = {
    "info": {
        "title": "HVAC Consumption API",
        "version": "1.0"
    },
    "tags": [
        {
            "name": "prediction",
            "description": "Operatii legate de predictii HVAC"
        },
        {
            "name": "real-consumption",
            "description": "Operatii legate de consumul real"
        },
        {
            "name": "building",
            "description": "Operatii legate de cladirile din dataset"
        },
        {
            "name": "baseline",
            "description": "Operatii legate de baseline-urile din dataset"
        },

    ]
}
