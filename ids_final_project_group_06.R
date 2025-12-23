library(dplyr)
library(tidytext)
library(textstem)
library(textclean)
library(SnowballC)
library(tokenizers)
library(readxl)
library(ggplot2)
library(text2vec)
library(irlba)
library(Matrix)
library(dbscan)
library(cluster)
library(stringr)
library(tidyr)
library(scales)
library(rlang)
setwd("E:/ids_final_project_group_06/")

dataset <- read_excel("ids_final_dataset_sample_group_06.xlsx", sheet = "train_40k")
dataset <- dataset %>% 
  slice(1:10000) %>%   
  select(productId, Title, reviews, Score)
View(dataset)
dataset$reviews <- as.character(dataset$reviews)
dataset$reviews[is.na(dataset$reviews)] <- ""
dataset$reviews <- tolower(dataset$reviews)
dataset$reviews <- replace_contraction(dataset$reviews)
dataset$reviews <- str_replace_all(dataset$reviews, "<.*?>", " ")
dataset$reviews <- str_replace_all(dataset$reviews, "[^a-z\\s]", " ")
dataset$reviews <- str_squish(dataset$reviews)
dataset$tokens <- tokenize_words(dataset$reviews)
dataset$tokens <- lapply(dataset$tokens, function(tokens) {
  tokens <- lemmatize_words(tokens)
  tokens <- wordStem(tokens, language = "en")
  tokens
})
default_stopwords <- stop_words$word
custom_words <- c("product","item","amazon","review","buy","this","be","a")
all_stopwords <- unique(c(default_stopwords, custom_words))
dataset$tokens <- lapply(dataset$tokens, function(tokens) tokens[!tokens %in% all_stopwords])
dataset$clean_reviews <- vapply(dataset$tokens, paste, collapse = " ", FUN.VALUE = character(1))
dataset <- dataset %>%
  mutate(sentiment = case_when(
    Score %in% c(1,2) ~ "Bad",
    Score == 3       ~ "Neutral",
    Score %in% c(4,5) ~ "Good",
    TRUE              ~ "Neutral"
  )) %>%
  mutate(sent_num = case_when(
    sentiment == "Good"    ~ 1,
    sentiment == "Neutral" ~ 0,
    sentiment == "Bad"     ~ -1,
    TRUE                   ~ 0
  ))
sentiment_summary <- dataset %>%
  group_by(sentiment) %>%
  summarise(count = n(), .groups = "drop")
ggplot(sentiment_summary, aes(x = sentiment, y = count, fill = sentiment)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = count), vjust = -0.5) +
  labs(title = "Sentiment Distribution Based on Review Score",
       x = "Sentiment Category", y = "Number of Reviews") +
  scale_fill_manual(values = c("Bad" = "red", "Neutral" = "gray", "Good" = "green")) +
  theme_minimal(base_size = 14)
tidy_reviews <- dataset %>%
  select(productId, clean_reviews) %>%
  unnest_tokens(word, clean_reviews) %>%
  filter(!word %in% all_stopwords)
word_counts <- tidy_reviews %>%
  count(productId, word, sort = TRUE)
tfidf_data <- word_counts %>%
  bind_tf_idf(word, productId, n) %>%
  arrange(desc(tf_idf))
top_products <- dataset %>%
  count(productId) %>%
  slice_max(n, n = 6) %>%
  pull(productId)
top_words_plot <- tfidf_data %>%
  filter(productId %in% top_products) %>%
  group_by(productId) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup()
ggplot(top_words_plot, aes(x = reorder_within(word, tf_idf, productId),
                           y = tf_idf, fill = productId)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~productId, scales = "free_y") +
  scale_x_reordered() +
  coord_flip() +
  labs(title = "Top 10 TF-IDF Words per Product (Top 6 Products)",
       x = "Words", y = "TF-IDF") +
  theme_minimal(base_size = 14)
txt <- dataset$clean_reviews
it <- itoken(txt, tokenizer = word_tokenizer, progressbar = TRUE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab,
                          term_count_min = 5,
                          doc_proportion_max = 0.5,
                          doc_proportion_min = 0.001)
vec <- vocab_vectorizer(vocab)
dtm <- create_dtm(it, vec)
tfidf <- TfIdf$new()
X_tfidf <- tfidf$fit_transform(dtm)
max_possible <- min(dim(X_tfidf)) - 1
lsa_dims <- min(100, max(2, max_possible))
svd_fit <- irlba::irlba(X_tfidf, nv = lsa_dims)
X_lsa <- svd_fit$u %*% diag(svd_fit$d)
num_feats <- as.matrix(dataset[, c("Score", "sent_num")])
features <- cbind(X_lsa, num_feats)
features_scaled <- scale(features)
set.seed(123)
pick_k_sil <- function(Xscaled, k_min = 2, k_max = 8) {
  uniq_n <- nrow(unique(as.data.frame(Xscaled)))
  k_max <- max(k_min, min(k_max, uniq_n - 1))
  best_k <- k_min; best_sil <- -Inf
  dmat <- dist(Xscaled)
  for (k in k_min:k_max) {
    km <- kmeans(Xscaled, centers = k, nstart = 25, iter.max = 200)
    sil <- cluster::silhouette(km$cluster, dmat)
    avg <- mean(sil[, "sil_width"])
    if (avg > best_sil) { best_sil <- avg; best_k <- k }
  }
  best_k
}
k <- pick_k_sil(features_scaled, 2, 8)
cat("Picked K via silhouette:", k, "\n")
km <- kmeans(features_scaled, centers = k, nstart = 50, iter.max = 300)
dataset$cluster_kmeans <- paste0("K", km$cluster)
dmat <- dist(features_scaled)
hc <- hclust(dmat, method = "ward.D2")
hc_cut <- cutree(hc, k = k)
dataset$cluster_hclust <- paste0("H", hc_cut)
minPts <- max(5, round(ncol(features_scaled)/10))
knn_d <- dbscan::kNNdist(features_scaled, k = minPts)
eps <- as.numeric(quantile(knn_d, probs = 0.90, na.rm = TRUE))
db <- dbscan::dbscan(features_scaled, eps = eps, minPts = minPts)
dataset$cluster_dbscan <- ifelse(db$cluster == 0, "Noise", paste0("D", db$cluster))
pca <- prcomp(features_scaled, center = TRUE, scale. = FALSE)
vis <- as.data.frame(pca$x[,1:2])
names(vis) <- c("X1","X2")
plot_clusters <- function(df2d, labels, title) {
  ggplot(data.frame(df2d, cl = factor(labels)), aes(X1, X2, color = cl)) +
    geom_point(size = 2.2, alpha = 0.85) +
    guides(color = guide_legend(override.aes = list(size = 4, alpha = 1))) +
    theme_minimal(base_size = 14) +
    labs(title = title, x = "Component 1", y = "Component 2", color = "Cluster")
}
print(plot_clusters(vis, dataset$cluster_kmeans, "K-means clusters (PCA view)"))
print(plot_clusters(vis, dataset$cluster_hclust, "Hierarchical clusters (PCA view)"))
print(plot_clusters(vis, dataset$cluster_dbscan, "DBSCAN clusters (PCA view)"))
plot_top_words_per_cluster <- function(cluster_col, dataset, top_n = 10, title_prefix = "") {
  cluster_col_q <- enquo(cluster_col)
  cluster_reviews <- dataset %>%
    select(!!cluster_col_q, clean_reviews, sentiment) %>%
    rename(cluster = !!cluster_col_q)
  top_clusters <- cluster_reviews %>%
    filter(cluster != "Noise") %>%
    count(cluster) %>%
    slice_max(n, n = 6) %>%
    pull(cluster)
  cluster_reviews <- cluster_reviews %>%
    filter(cluster %in% top_clusters)
  tidy_cluster <- cluster_reviews %>%
    unnest_tokens(word, clean_reviews) %>%
    filter(!word %in% all_stopwords)
  word_counts_cluster <- tidy_cluster %>%
    count(cluster, word, sort = TRUE)
  tfidf_cluster <- word_counts_cluster %>%
    bind_tf_idf(word, cluster, n) %>%
    arrange(cluster, desc(tf_idf))
  top_words_cluster <- tfidf_cluster %>%
    group_by(cluster) %>%
    slice_max(tf_idf, n = top_n) %>%
    ungroup()
  p <- ggplot(top_words_cluster, aes(x = reorder_within(word, tf_idf, cluster),
                                     y = tf_idf, fill = cluster)) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~cluster, scales = "free_y") +
    scale_x_reordered() +
    coord_flip() +
    labs(title = paste(title_prefix, "- Top", top_n, "TF-IDF Words per Cluster (Top 6)"),
         x = "Words", y = "TF-IDF") +
    theme_minimal(base_size = 14)
  print(p)
  sentiment_summary <- cluster_reviews %>%
    group_by(cluster, sentiment) %>%
    summarise(count = n(), .groups = "drop") %>%
    group_by(cluster) %>%
    mutate(percent = round(100 * count / sum(count), 2))
  p2 <- ggplot(sentiment_summary, aes(x = cluster, y = percent, fill = sentiment)) +
    geom_bar(stat = "identity", position = "stack") +
    labs(title = paste(title_prefix, "- Sentiment Distribution per Cluster"),
         x = "Cluster", y = "Percentage (%)") +
    theme_minimal(base_size = 14)
  print(p2)
}
plot_top_words_per_cluster(cluster_kmeans, dataset, 10, "K-means")
plot_top_words_per_cluster(cluster_hclust, dataset, 10, "Hierarchical")
plot_top_words_per_cluster(cluster_dbscan, dataset, 10, "DBSCAN")