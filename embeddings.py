import torch
import torch.nn as nn
import torch.nn.functional as F


class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_rate = config.num_rate
        self.num_genre = config.num_genre
        self.num_director = config.num_director
        self.num_actor = config.num_actor
        self.embedding_dim = config.embedding_dim

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate, 
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_gender = config.num_gender
        self.num_age = config.num_age
        self.num_occupation = config.num_occupation
        self.num_zipcode = config.num_zipcode
        self.embedding_dim = config.embedding_dim

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )
import torch
import torch.nn as nn
import torch.nn.functional as F


class item_movielens(torch.nn.Module):
    def __init__(self, config):
        super(item_movielens, self).__init__()
        self.num_rate = config.num_rate-1
        self.num_genre = config.num_genre-1
        self.num_director = config.num_director-1
        self.num_actor = config.num_actor-1
        self.embedding_dim = config.embedding_dim

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate, 
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)


class user_movielens(torch.nn.Module):
    def __init__(self, config):
        super(user_movielens, self).__init__()
        self.num_gender = config.num_gender-1
        self.num_age = config.num_age-1
        self.num_occupation = config.num_occupation-1
        self.num_zipcode = config.num_zipcode-1
        self.embedding_dim = config.embedding_dim

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)

class item_yelp(torch.nn.Module):
    def __init__(self, config):
        # stars, postalcode, reviewcount, city, state
        super(item_yelp, self).__init__()
        self.num_stars = config.num_stars
        self.num_postalcode = config.num_postalcode
        self.embedding_dim = config.embedding_dim

        self.embedding_stars = torch.nn.Embedding(
            num_embeddings=self.num_stars,
            embedding_dim=self.embedding_dim
        )
        self.embedding_postalcode = torch.nn.Embedding(
            num_embeddings=self.num_postalcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, stars_idx, postalcode_idx):
        stars_emb = self.embedding_stars(stars_idx)
        postalcode_emb = self.embedding_postalcode(postalcode_idx)

        return torch.cat((stars_emb, postalcode_emb), 1)

class user_yelp(torch.nn.Module):
    def __init__(self, config):
        super(user_yelp, self).__init__()
        self.num_fans = config.num_fans
        self.num_avgrating = config.num_avgrating
        self.embedding_dim = config.embedding_dim

        self.embedding_fans = torch.nn.Embedding(
            num_embeddings=self.num_fans,
            embedding_dim=self.embedding_dim
        )
        self.embedding_avgrating = torch.nn.Embedding(
            num_embeddings=self.num_avgrating,
            embedding_dim=self.embedding_dim
        )

    def forward(self, fans_idx, avgrating_idx):
        fans_emb = self.embedding_fans(fans_idx)
        avgrating_emb = self.embedding_avgrating(avgrating_idx)
        return torch.cat((fans_emb, avgrating_emb), 1)


    
class item_dbook(torch.nn.Module):
    def __init__(self, config):
        super(item_dbook, self).__init__()
        self.num_publisher = config.num_publisher -1 
        self.embedding_dim = config.embedding_dim

        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=self.num_publisher, 
            embedding_dim=self.embedding_dim
        )

    def forward(self, publisher_idx, vars=None):
        publisher_emb = self.embedding_publisher(publisher_idx)
        return publisher_emb

class user_dbook(torch.nn.Module):
    def __init__(self, config):
        super(user_dbook, self).__init__()
        self.num_location = config.num_location - 1
        self.embedding_dim = config.embedding_dim

        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )


    def forward(self, location_idx):
        location_emb = self.embedding_location(location_idx)
        return location_emb
        
class item_lastfm(torch.nn.Module):
    def __init__(self, config):
        super(item_lastfm, self).__init__()
        self.num_lastfm_item = config.num_lastfm_item
        self.embedding_dim = config.embedding_dim

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.num_lastfm_item, 
            embedding_dim=self.embedding_dim
        )

    def forward(self, idx, vars=None):
        publisher_emb = self.embedding(idx)
        return publisher_emb

class user_lastfm(torch.nn.Module):
    def __init__(self, config):
        super(user_lastfm, self).__init__()
        self.num_lastfm_user = config.num_lastfm_user
        self.embedding_dim = config.embedding_dim

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.num_lastfm_user,
            embedding_dim=self.embedding_dim
        )


    def forward(self, idx):
        location_emb = self.embedding(idx)
        return location_emb
