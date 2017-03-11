class Business(object):

    def __init__(self, features):
        self.feature_dict = features

    def handle_city(self):
        del self.feature_dict["city"]

    def handle_neighborhood(self):
        del self.feature_dict["neighborhood"]

    def handle_name(self):
        del self.feature_dict["name"]

    def handle_ID(self):
        del self.feature_dict["business_id"]

    def handle_lon(self):
        self.feature_dict["longitude"] = int(self.feature_dict["longitude"])

    def handle_hours(self):
        del self.feature_dict["hours"]

    def handle_state(self):
        del self.feature_dict["state"]

    def handle_postal_code(self):
        del self.feature_dict["postal_code"]

    def handle_categories(self):
        del self.feature_dict["categories"]

    def handle_stars(self):
        self.feature_dict["stars"] = int(self.feature_dict["stars"])

    def handle_address(self):
        del self.feature_dict["address"]

    def handle_lat(self):
        self.feature_dict["latitude"] = int(self.feature_dict["latitude"])

    def handle_review_count(self):
        self.feature_dict["review_count"] = int(self.feature_dict["review_count"])

    def handle_attrs(self):
        del self.feature_dict["attributes"]

    def handle_type(self):
        del self.feature_dict["type"]

    def handle_isopen(self):
        self.feature_dict["latitude"] = int(self.feature_dict["latitude"])

    def featurize(self):
        for k in self.feature_dict.keys():
            k = str(k)
            if k == "city":
                self.handle_city()
            elif k == "neighborhood":
                self.handle_neighborhood()
            elif k == "name":
                self.handle_name()
            elif k == "business_id":
                self.handle_ID()
            elif k == "longitude":
                self.handle_lon()
            elif k == "hours":
                self.handle_hours()
            elif k == "state":
                self.handle_state()
            elif k == "postal_code":
                self.handle_postal_code()
            elif k == "categories":
                self.handle_categories()
            elif k == "stars":
                self.handle_stars()
            elif k == "address":
                self.handle_address()
            elif k == "latitude":
                self.handle_lat()
            elif k == "review_count":
                self.handle_review_count()
            elif k == "attributes":
                self.handle_attrs()
            elif k == "type":
                self.handle_type()
            elif k == "is_open":
                self.handle_isopen()
            else:
                return AssertionError("Unexpected field in data: ", k)

        # sort values in alphabetical order of their keys
        sorted_kvs = sorted(self.feature_dict.items())
        return [v for (k, v) in sorted_kvs]


