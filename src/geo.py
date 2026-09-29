"""Region (state / province) vocabulary used to pull region components out of addresses.

Written from general knowledge of US state and Indian state/UT names and their postal
codes; names and codes both map to the same canonical token, so "Ohio" == "OH" and
"Uttar Pradesh" == "UP". Codes are only matched as a *whole* address component, never
as a free token. Native-script region names are not listed here: they are learned from
the training pairs (see normalize.learn_region_map). Unknown countries simply get no
canonicalisation and fall back to plain token matching.
"""

US = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}
INDIA = {
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "ct": "cg", "goa": "goa", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "ts": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "ut": "uk", "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk", "ladakh": "lad",
    "chandigarh": "chd", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "lakshadweep": "ld",
}

# French regions, and departments mapped to their region (so "Nord" == "Hauts-de-France").
FRANCE_REGIONS = {
    "hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
    "ile de france": "idf", "auvergne rhone alpes": "ara", "provence alpes cote d azur": "paca",
    "occitanie": "occ", "grand est": "ges", "bretagne": "bre", "normandie": "nor",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "corse": "cor",
}
FRANCE_DEPTS = {
    "hdf": ["nord", "pas de calais", "somme", "oise", "aisne"],
    "naq": ["gironde", "landes", "pyrenees atlantiques", "dordogne", "lot et garonne",
            "charente", "charente maritime", "deux sevres", "vienne", "haute vienne",
            "creuse", "correze"],
    "pdl": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "idf": ["seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis",
            "val de marne", "val d oise"],
    "bre": ["finistere", "morbihan", "ille et vilaine", "cotes d armor"],
    "ara": ["rhone", "isere", "loire", "savoie", "haute savoie", "ain", "puy de dome"],
    "paca": ["bouches du rhone", "var", "alpes maritimes", "vaucluse"],
    "occ": ["haute garonne", "herault", "gard", "pyrenees orientales"],
    "ges": ["bas rhin", "haut rhin", "moselle", "marne"],
    "nor": ["seine maritime", "calvados", "manche", "eure", "orne"],
}
FRANCE = dict(FRANCE_REGIONS)
for code, depts in FRANCE_DEPTS.items():
    FRANCE.update({d: code for d in depts})


def _build():
    m = {}
    for table in (US, INDIA, FRANCE):
        for name, code in table.items():
            m[name] = code
            m.setdefault(code, code)
    return m


# component text (lowercase ascii) -> canonical region token (without the "@" prefix)
REGION = _build()

# city spelling variants -> one canonical token
CITY_SYN = {
    "bombay": "mumbai", "bengaluru": "bangalore", "calcutta": "kolkata", "madras": "chennai",
    "gurugram": "gurgaon", "poona": "pune", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "baroda": "vadodara", "mysuru": "mysore", "benaras": "varanasi",
}
