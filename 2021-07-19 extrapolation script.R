# Do the extrapolations to the target regions.
#
# Two models
#     thaw_xgb_fit is the model that applies in thaw feature extent
#     CWPP_xgb_fit is the model for outside of thaw feature (in "other CWPP" extent)

#---------------------------------------------------------------



library(tidyverse)
library(xgboost)


# I don't want to see scientific notation
options(scipen=8) 

# Set random seed for reproducibility
set.seed(42)

#############################################################
# My functions
#############################################################

# FUNCTION create_dummy_vars
# PARAMETERS:
#   df : a data frame
#   factor_name : a character string that specified the factor-type
#         variable in df to be converted into dummy variables
# RETURNS:  modified df, now containing the dummy variables
create_dummy_vars <- function(df, factor_name) {
  factor_levels <- levels(df[[factor_name]])
  for (n in 1:length(factor_levels)) {
    newvar_name <- paste0(factor_name, as.character(n))
    df[[newvar_name]] = ifelse(is.na(df[[factor_name]]), 
                               NA,
                               ifelse(df[[factor_name]]==factor_levels[n],1,0))
    # remove the variable has no variance
    if (var(df[[newvar_name]], na.rm = TRUE)==0){
      df[[newvar_name]] <- NULL
    }
  }
  return(df)
}


###################################################################

# Load the data
load("../../Data/permafrost_df FNSB CWPP.RData")
load("xgboost FNSB-thaw extent.RData")
load("xgboost FNSB-Other CWPP.RData")

# Both models were trained on the data where "Water" was removed.  I will similarly
# distinquish "Water" from the "thaw feature" and "other CWPP" extents.

comb_df <- comb_df %>%
  mutate(which_extent_w = ifelse(ABOVE_1984_adj=="Water", "Water", as.character(which_extent))) %>%
  mutate(which_extent_w = as.factor(which_extent_w))



# To use xgboost models, need to do the following:

# Change all the predictive factors to numeric variables,
# either into numeric equivalents or into hotcoding, in the same way as
# for the data the models were trained on.


comb_df <- comb_df %>%
  mutate(ice_wedge = ifelse(thaw_feature == "ice wedge", 1, 0))%>%
  mutate(ice_wedge = as.factor(ice_wedge)) %>%
  select(XCoord, YCoord,  # x-y coordinates
         which_extent_w,    # ground ice, thaw feature, or other CWPP region -- or water.
         ground_ice_3,    # labels to extrapolate, available only in the "ground ice" extent.
         ice_wedge,       # whether there is an "ice wedge" thaw feature.  Available only in thaw feature extent.    
         Tanana_floodplain,         # indicator for the river Tanana floodplain
         ELEV, Local_Elev, SLOPE,   # Relevant topography.  Will need to decide if elevation (ELEV) remains relevant.
         S_TEMP_FNSB_1996TO2005, W_TEMP_FNSB_1996TO2005,  # average decadal temperature
         isWetland,                 # wetland indicator
         ABOVE_1984_adj             # vegetation type
  ) %>%
  rename(temp_summer = S_TEMP_FNSB_1996TO2005) %>%
  rename(temp_winter = W_TEMP_FNSB_1996TO2005)

#........................................................................

# Filter thaw extent without water, and drop "Water" as factor type
thaw_df <- comb_df %>%
  filter(which_extent_w == "thaw feature") %>%
  select(-which_extent_w, -ground_ice_3) %>%
  mutate(ABOVE_1984_adj = droplevels(ABOVE_1984_adj))

# Select only observations with complete records (12 observations get dropped)
thaw_df <- thaw_df[complete.cases(thaw_df),]

# Change binary labels.  
thaw_df <- thaw_df %>% 
  mutate(ice_wedge = as.integer(ice_wedge)-1 ) %>%
  mutate(Tanana_floodplain = as.integer(Tanana_floodplain)-1 ) %>%
  mutate(isWetland = as.integer(isWetland)-1 ) 


# Hotcode the vegetation data
thaw_df <- create_dummy_vars(thaw_df, "ABOVE_1984_adj")
thaw_df <- select(thaw_df, -ABOVE_1984_adj)


#...................................................

# Filter other CWPP extent without water, and drop "Water" as factor type
CWPP_df <- comb_df %>%
  filter(which_extent_w == "other CWPP") %>%
  select(-which_extent_w, -ground_ice_3, -ice_wedge) %>%
  mutate(ABOVE_1984_adj = droplevels(ABOVE_1984_adj))

# Select only observations with complete records (9097 observations get dropped)
CWPP_df <- CWPP_df[complete.cases(CWPP_df),]

# Change binary labels.  
CWPP_df <- CWPP_df %>% 
  mutate(Tanana_floodplain = as.integer(Tanana_floodplain)-1 ) %>%
  mutate(isWetland = as.integer(isWetland)-1 ) 


# Hotcode the vegetation data
CWPP_df <- create_dummy_vars(CWPP_df, "ABOVE_1984_adj")
CWPP_df <- select(CWPP_df, -ABOVE_1984_adj)

#......................................................................
# For completeness, do ground ice extent (minus "Water")

ice_df <- comb_df %>%
  filter(which_extent_w == "ground ice") %>%
  select(-which_extent_w, -ground_ice_3) %>%
  mutate(ABOVE_1984_adj = droplevels(ABOVE_1984_adj))

# Select only observations with complete records (12 observations get dropped)
ice_df <- ice_df[complete.cases(ice_df),]

# Change binary labels.  
ice_df <- ice_df %>% 
  mutate(ice_wedge = as.integer(ice_wedge)-1 ) %>%
  mutate(Tanana_floodplain = as.integer(Tanana_floodplain)-1 ) %>%
  mutate(isWetland = as.integer(isWetland)-1 ) 


# Hotcode the vegetation data
ice_df <- create_dummy_vars(ice_df, "ABOVE_1984_adj")
ice_df <- select(ice_df, -ABOVE_1984_adj)


#...................................................

#...................................................
# Extrapolate to thaw feature extent

thaw_DMatrix <- xgb.DMatrix(data=as.matrix(select(thaw_df, -XCoord, -YCoord)))

# predict ground ice labels
ground_ice_3_prob_output <- predict(thaw_xgb_fit, newdata = thaw_DMatrix)

# The output is an array that is three times the length of the data.
# Sort out the output by type of ground_ice_3 labels
ground_ice_3_probs <- matrix(ground_ice_3_prob_output, 
                             nrow=3, # three classes 
                             ncol = length(ground_ice_3_prob_output)/3)

load("./model stats/thaw_xgb_fit cutoff maximizing F1 for High.RData")

rm(ground_ice_3_prob_output)

ground_ice_3_probs <- ground_ice_3_probs %>% 
  t() %>%  # transpose
  as.data.frame()

names(ground_ice_3_probs) <- c("Low_prob", "Medium_prob", "High_prob")

ground_ice_3_probs <- ground_ice_3_probs %>%
  mutate(ground_ice_pred = ifelse(High_prob >= cutoff, "High",  # call this observation "High", even if something else has higher prob
                                  ifelse(High_prob >= Medium_prob & High_prob >= Low_prob, "High",  # if High_prob is the highest of the three
                                         ifelse(Low_prob >= High_prob & Low_prob >= Medium_prob, "Low", "Medium")))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))

thaw_df <- cbind(thaw_df, ground_ice_3_probs)

# Keep only the x-y coordinates and the extrapolated info
thaw_df <- thaw_df %>%
  select(XCoord, YCoord, Low_prob, Medium_prob, High_prob, ground_ice_pred)

thaw_cutoff <- cutoff

rm(thaw_DMatrix)

#...................................................
# Extrapolate to the rest of CWPP extent

CWPP_DMatrix <- xgb.DMatrix(data=as.matrix(select(CWPP_df, -XCoord, -YCoord)))

# predict ground ice labels
ground_ice_3_prob_output <- predict(CWPP_xgb_fit, newdata = CWPP_DMatrix)

# The output is an array that is three times the length of the data.
# Sort out the output by type of ground_ice_3 labels
ground_ice_3_probs <- matrix(ground_ice_3_prob_output, 
                             nrow=3, # three classes 
                             ncol = length(ground_ice_3_prob_output)/3)

load("./model stats/CWPP_xgb_fit cutoff maximizing F1 for High.RData")

rm(ground_ice_3_prob_output)

ground_ice_3_probs <- ground_ice_3_probs %>% 
  t() %>%  # transpose
  as.data.frame()

names(ground_ice_3_probs) <- c("Low_prob", "Medium_prob", "High_prob")

ground_ice_3_probs <- ground_ice_3_probs %>%
  mutate(ground_ice_pred = ifelse(High_prob >= cutoff, "High",  # call this observation "High", even if something else has higher prob
                                  ifelse(High_prob >= Medium_prob & High_prob >= Low_prob, "High",  # if High_prob is the highest of the three
                                         ifelse(Low_prob >= High_prob & Low_prob >= Medium_prob, "Low", "Medium")))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))

CWPP_df <- cbind(CWPP_df, ground_ice_3_probs)

# Keep only the x-y coordinates and the extrapolated info
CWPP_df <- CWPP_df %>%
  select(XCoord, YCoord, Low_prob, Medium_prob, High_prob, ground_ice_pred)

CWPP_cutoff <- cutoff

rm(CWPP_DMatrix, CWPP_xgb_fit)

#...................................................
# For completeness, "Extrapolate" to ground ice extent

ice_DMatrix <- xgb.DMatrix(data=as.matrix(select(ice_df, -XCoord, -YCoord)))

# predict ground ice labels (using thaw_xgb_fit model)
ground_ice_3_prob_output <- predict(thaw_xgb_fit, newdata = ice_DMatrix)

# The output is an array that is three times the length of the data.
# Sort out the output by type of ground_ice_3 labels
ground_ice_3_probs <- matrix(ground_ice_3_prob_output, 
                             nrow=3, # three classes 
                             ncol = length(ground_ice_3_prob_output)/3)


rm(ground_ice_3_prob_output)

ground_ice_3_probs <- ground_ice_3_probs %>% 
  t() %>%  # transpose
  as.data.frame()

names(ground_ice_3_probs) <- c("Low_prob", "Medium_prob", "High_prob")


# Use thaw_cutoff
ground_ice_3_probs <- ground_ice_3_probs %>%
  mutate(ground_ice_pred = ifelse(High_prob >= thaw_cutoff, "High",  # call this observation "High", even if something else has higher prob
                                  ifelse(High_prob >= Medium_prob & High_prob >= Low_prob, "High",  # if High_prob is the highest of the three
                                         ifelse(Low_prob >= High_prob & Low_prob >= Medium_prob, "Low", "Medium")))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))

ice_df <- cbind(ice_df, ground_ice_3_probs)

# Keep only the x-y coordinates and the extrapolated info
ice_df <- ice_df %>%
  select(XCoord, YCoord, Low_prob, Medium_prob, High_prob, ground_ice_pred)

rm(ice_DMatrix, cutoff, thaw_cutoff, CWPP_cutoff)
rm(thaw_xgb_fit, ground_ice_3_probs)


#####################################################################
# Recombine the info into comb_df


ext_df <- rbind(ice_df, thaw_df, CWPP_df)
comb_df <- merge(comb_df, ext_df, by = c("XCoord", "YCoord"), all.x = TRUE, all.y = FALSE)
rm(ext_df)

save(comb_df, file = "Extended ground ice data.RData")

#####################################################################

# get original ground ice labels
comb_dfx <- comb_df
load("../../Data/permafrost_df FNSB CWPP.RData")
orig_ice_df <- comb_df %>% select(XCoord, YCoord, ground_ice_value)
rm(comb_df)
comb_df <- comb_dfx
rm(comb_dfx)
comb_df <- merge(comb_df, orig_ice_df, by = c("XCoord", "YCoord"), all.x = TRUE, all.y = FALSE)
rm(orig_ice_df)
save(comb_df, file = "Extended ground ice data.RData")


comb_df <- comb_df %>%
  mutate(which_extent = ifelse(which_extent_w == "ground ice" & ground_ice_value == "water_body",
                               "Water", as.character(which_extent_w))) %>%
  mutate(which_extent = as.factor(which_extent)) %>%
  mutate(which_extent = factor(which_extent, levels = c("ground ice", "thaw feature", "other CWPP", "Water")))



comb_df <- comb_df %>%
  mutate(ground_ice_extended = ifelse(which_extent=="Water", "Water",
                                   ifelse(which_extent=="ground ice", as.character(ground_ice_3),
                                          as.character(ground_ice_pred)))) %>%
  mutate(ground_ice_extended = as.factor(ground_ice_extended)) %>%
  mutate(ground_ice_extended = factor(ground_ice_extended, levels = c("Water", "Low", "Medium", "High")))

save(comb_df, file = "Extended ground ice data.RData")

ground_ice_df <- comb_df %>%
  select(XCoord, YCoord, ground_ice_value, ground_ice_extended, Low_prob, Medium_prob, High_prob) %>%
  filter(!is.na(ground_ice_extended))

write.csv(ground_ice_df, file="FNSB_CWPP_ground_ice_extended.csv", row.names = FALSE)

###############################################################################################

# for peeking at the graphs
smp_size <- floor(0.05*nrow(comb_df))
small_df <- comb_df[sample(seq_len(nrow(comb_df)), size = smp_size), ]

# Visualize the results for extended ground ice categories
plt <- ground_ice_df  %>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_tile(aes(fill=ground_ice_extended)) + 
  scale_fill_manual(values 
                    = c("Water" = "blue",   
                        "Low" = "green",
                        "Medium" = "yellow",
                        "High" = "red"
                    )) 


ggsave(plt, file="./Graphs/CWPP ground_ice_extended.png", height = 5, width = 7)


# visualise the probabilities
plt <- comb_df  %>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_tile(aes(fill=High_prob)) + 
  scale_fill_distiller(palette = "RdYlBu")

ggsave(plt, file="./Graphs/CWPP High_prob.png", height = 5, width = 7)

plt <- comb_df  %>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_tile(aes(fill=Medium_prob)) + 
  scale_fill_distiller(palette = "RdYlBu")

ggsave(plt, file="./Graphs/CWPP Medium_prob.png", height = 5, width = 7)

plt <- comb_df  %>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_tile(aes(fill=Low_prob)) + 
  scale_fill_distiller(palette = "RdYlBu")

ggsave(plt, file="./Graphs/CWPP Low_prob.png", height = 5, width = 7)


# # Visualize missing values
# plt <- comb_df  %>%
#   mutate(missing_ice_values = ifelse(is.na(ground_ice_extended), "missing", "present")) %>%
#   mutate(missing_ice_values = as.factor(missing_ice_values)) %>%
#   ggplot(aes(x=XCoord,
#              y=YCoord)) +
#   geom_tile(aes(fill=missing_ice_values)) +
#   scale_fill_manual(values = c("missing" = "red", "present"="grey"))
# 
# ggsave(plt, file="./Graphs/missing ice values.png", height = 5, width = 7)
  