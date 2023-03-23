
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
import pandas as pd

# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.
#pvt = df.pivot_table(index="SepetId", values="CreateDate", columns="Hizmet", aggfunc="count").applymap(lambda x:1 if x>0 else 0)

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Datasets/movie.csv')
    rating = pd.read_csv('Datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId") #Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index #Adım 3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating") #Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz
    return user_movie_df

user_movie_df = create_user_movie_df()

user_movie_df.head()


#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
# random_useri user_movie_df den seç çünkü bunun indexinde kullanıcılar vardı. rastgele 1 tane örneklem seç deriz.
# random_state=45 seçildi. VK ile aynı sonuçları alabilmek adına. aynı kullanıcı ID si seçmek için.  .values ile de integera çerilecek.

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# random_user 28941 ID li user'mış. Amaç bu kullanıcı ne izlemiş onu bulmak ve sonrada bu kullanıcıya benzer kişileri bulmak.



# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

# user_ıdler üzerinden ramdom_user'in izlediği filmleri seçicez.
random_user_df = user_movie_df[user_movie_df.index == random_user] # sütunlarda bir seçim yapmadığım için yine tüm filmler geldi
# çıktıdaki NaN ifadeleri kullancının o filme puan vermediğini ifade ediyor. dolu ise izlemiş ve puan vermiş
# Nan olmayanları görmemiz gerekiyor.

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
# random_user_df.columns sütunlarda gez.[random_user_df.notna().any()], notna().any(), NaN ifade mi diye sor. liste olarak gelsin

# ilk 3 gözlem:
#Out[11]:
#['Ace Ventura: Pet Detective (1994)',
# 'Ace Ventura: When Nature Calls (1995)',
# 'Aladdin (1992)',

# çıktıda sadece sütun bilgileri geldi. kullanıcının bu filmleri izlediğini nasıl doğrulayailiriz?
#satır-sütunlarda seçim yap.
# user_movie_df.index == random_user >> user_movie_df inde random_user in indexini bul
#ve user_movie_df.columns == "Silence of the Lambs, The (1991)"user_movie_df sütunlarında Silence of the Lambs, şu filmi bul.
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == 'Ace Ventura: Pet Detective (1994)']

#title    Ace Ventura: Pet Detective (1994)
#userId
#28941.0                                3.0 #Yukarıdaki çıktıda izlediği filmler arasından ilkni doğrulamak amaçlı kaç puan verdiğini sordum.
#kullanıcı bu filmi izlemiş ve 3 puan vermiş.


# 28941.0 ıd li kullanıcı kaç film izlemiş?
len(movies_watched)
# Out[14]: 33 film izlemiş

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
movies_watched_df = user_movie_df[movies_watched] #sadece izlenen filmlere indirgemek istedim.
#[138493 rows x 33 columns] 33 film izlemişti. çıktı doğru.


# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.

user_movie_count = movies_watched_df.T.notnull().sum() #Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisi

user_movie_count = user_movie_count.reset_index()  #user_ıd leri index olmaktan kurtar

user_movie_count.columns = ["userId", "movie_count"] # columnslerin ismini değiştir.
# userId     movie_count
# 138493.0        9

# çıktı sonucunu yorumlarsak: son index için: 138493 userıd li kullanıcı benim seçtiğim 28941 idli kullanıcının izlediği filmlerden 9 tanesini izlemiş.



# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

33*0.60 #33 filmin %60'ı = 19.8 en azından seçilen kullanıcı ile ortak 20 film ve üstünü izleyenleri getir.
#perc = len(movies_watched) * 60 / 100 diğer yol

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# [3202 rows x 2 columns] 28941 idli kullanıcı ile ortak filmleri izleyen 138bin kullanıcı vardı. bu seçimden sonra 3202'e düştü.

# bunların user_id'lerinin çıktısını istiyorum?
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]


#28941.0 ıd li kullanıcının izlediği filmlerin hepsini izleyen kaç kullanııcı vardır?
user_movie_count[user_movie_count["movie_count"] == 33].count()
# 17 kullanıcı varmış.



#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]]) # 28941.0 ıd li kullanıcı ya göre oluşturdugumuz user_same_movies'e göre (bu random_user_df ti) izlenen filmlere göre birleştir.


# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

# final_df, userID leri değişken olarak atamalıyız ki kullanıcıların birbirleriyle olan korelasyonlarına bakabilelim.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates() # burdaki çıktıda okunma problemi var. klasik df e çevirelim.

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2'] # indexlerde yer alan isimlendirmeleri düzenle

corr_df = corr_df.reset_index()
corr_df

# tüm kullanıcıların birbirleriyle olan korelasyonu.
#        user_id_1  user_id_2      corr
#0          28866.0    67756.0 -0.936065
#1          60562.0    37121.0 -0.915003
#2          34103.0    80593.0 -0.898718
#3          62575.0    21398.0 -0.896612


# 28941.0 ıd li kullanıcı ya göre yüksek korelasyonlu kullanıcıları istiyorum.





# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

#corr_df[corr_df["user_id_1"] == random_user] #user_id_1'ye random olarak belirlediğim 28941 id li kullanıcıyı ata.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

#   user_id_2      corr
#27    28941.0  1.000000 # bu kullanıcının kendisiyle olan ilişkisi.
#26    45158.0  0.800749 # 45158 idli kullanıcı ile pozitif yönlü kuvvetli ilişki var.
#25   101628.0  0.790405


top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Random belirdiğimiz 28941 kullanıcı ile aynı filmleri izlemiş benzer kullanıcıların kor. ulaştık ama bu kullanıcıların puan bilgileri yok

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz

rating = pd.read_csv('Datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner') #top_user ve ratingi birleştir.
#userId", "movieId", "rating bu değişkenlri istiyorum.

#Out[52]:
#         userId      corr  movieId  rating
#0       28941.0  1.000000        7     5.0
#1       28941.0  1.000000       11     3.0 çıktıda kullanıcının kendisiyle olan ilişkisinin korelasyonları da var.bunları çıkarıyorum.
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#Out[54]:
#        userId      corr  movieId  rating
#33      45158.0  0.800749        1     1.5 # belirlediğim random kullanıcı ile 45158 idli kullanıcı arasında pozitif yönlü 0.8 şiddetinde bir ilişki var.
#34      45158.0  0.800749        3     1.0

#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

# ccor ve rating etkisini aynı anda gözlemleyeceğim. ratingleri ccor'a göre düzeltmiş olucam.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

#Out[58]: sadece reytinglere göre önerede bulunursak korelasyonlar göz ardı edilecekti. ratingleri corr değerlere göre  düzeltmiş olduk.
#         userId      corr  movieId  rating  weighted_rating
#33      45158.0  0.800749        1     1.5         1.201124
#34      45158.0  0.800749        3     1.0         0.800749



# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()
#        weighted_rating
#movieId
#1               2.205824
#2               1.463950


# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]

# df e çevir.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)



# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.

# bu filmler hangileri?
movie = pd.read_csv('datasets/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]



#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.

movie = pd.read_csv('Datasets/movie.csv')
rating = pd.read_csv('Datasets/rating.csv')

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending = False)["movieId"]
#movie_id
#Out[112]:
#15643060    7044 #öneri yapılacak kullanının 5 puan verdiği filmlerden en güncel olan film 7044 id li filmmiş.
#15642888      25
#15643008    2871
#15642976    1732

movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending = False)["movieId"][0:1].values[0]

#movie_id
#Out[108]: 7044

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
#movie[movie["movieId"] == 7044]["title"]
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_name = movie[movie["movieId"] == movie_id]["title"].values[0]



#'Wild at Heart (1990)' # 7044 idli filmin ismi imiş.

movie_name = user_movie_df[movie_name]



# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)

#Out[133]:
#title
#Wild at Heart (1990)                     1.000000
#My Science Project (1985)                0.570187  # bu filmin Wild at Heart ile pozitif yönlü orta şiddetli bir ilişkisi varmış.
#Mediterraneo (1991)                      0.538868
#old Man and the Sea, The (1958)          0.536192

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.

movies_from_item_based[1:6].index.to_list()


#Out[134]: Index(['My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea, The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']


