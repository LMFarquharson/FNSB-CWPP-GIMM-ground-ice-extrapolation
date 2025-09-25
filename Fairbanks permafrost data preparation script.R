# Fairbanks permafrost
# Data preparation script
# 5-June-2021

library(tidyverse)


# vegetation data
# In this dataset, the variables "ground_ice" and "all_thaw_feature" are screwed up,
# but the vegetation data is good.
veg_df <- read.csv("permafrost_data.txt", header = TRUE, na.strings = "NULL")

veg_df <- veg_df %>% select(-grid_code, -ground_ice, -all_thaw_feature)

# Info on the dataset from Monika:
#
# It should contain 10,237,600 rows which corresponds to the cwpp_veg raster cells. 
# Columns are x, y coordinates, grid_code (you can ignore, just an ID number), 
# FuelTypeCl (cwpp_veg type, you probably don't need it but I kept it in there just in case), 
# ground_ice, all_thaw_feature, FAI_VegPlus1984, FAI_VegPlus1994, FAI_VegPlus2004, 
# FAI_VegPlus2014 (FAI_VegPlus1984 to 2004 has the correction where some herbaceous 
# has been replaced with deciduous, 2014 is the same you used previously).
#
# Keep in mind that ground ice has data in only 916,285 cells and all_thaw_feature 
# has only 118,828 non-contiguous cells with data.

# > str(veg_df)
# 'data.frame':	10263234 obs. of  7 variables:
# $ XCoord         : num  327346 327406 327466 327496 327526 ...
# $ YCoord         : num  1612168 1612168 1612168 1612168 1612168 ...
# $ FuelTypeCl     : Factor w/ 24 levels "","Aquatic","Barren ground/gravel&dirt",..: 5 11 6 6 6 6 6 5 6 6 ...
# $ FAI_VegPlus1984: int  NA 11 11 6 11 11 11 11 11 11 ...
# $ FAI_VegPlus1994: int  NA 11 11 6 11 11 11 11 11 11 ...
# $ FAI_VegPlus2004: int  NA 11 11 6 4 4 4 4 16 16 ...
# $ FAI_VegPlus2014: int  NA 11 11 4 1 1 1 16 16 16 ...

summary(veg_df[,c(1,2)])

# > summary(veg_df[,c(1,2)])
# XCoord           YCoord       
# Min.   :     1   Min.   :      1  
# 1st Qu.:305836   1st Qu.:1648648  
# Median :338896   Median :1673308  
# Mean   :330613   Mean   :1668330  
# 3rd Qu.:361066   3rd Qu.:1697398  
# Max.   :392446   Max.   :1742818  
# NA's   :2        NA's   :2    

# The problem is that there are these weird small values in XCoord and YCoord

veg_df <- veg_df %>% 
  filter(XCoord > 1000 & YCoord > 1000) 

# # Look at one of the vegetation variables
# plt <- veg_df  %>%
#   ggplot(aes(x=XCoord,
#              y=YCoord)) +
#   geom_tile(aes(fill=FAI_VegPlus2014))

# rm(plt)


##################################################################################

# permafrost data
ice_df <- read.csv("permafrost_data2.txt", header = TRUE, na.strings = "NULL")

ice_df <- ice_df %>% select(-pointid)

# Info on the dataset from Monika:
#
# I reextracted only ground ice and and all-thaw-merged and now each column lists 
# the correct number of values (plus <Null>). 
# For ground_ice:
#   1 = low to moderate, 
#   2 = absent to low, 
#   3 = moderate to high, 
#   4 = high, 
#   5 = none, 
#   6 = low, 
#   7 = water_body 
# (clearly, the numbers do not imply quantity or are ordered correctly for Likert scale)
# For all_thaw_feature:
#   1 = thaw pond
#   2 = ice wedge
#   3 = thermokarst

# > str(ice_df)
# 'data.frame':	13307480 obs. of  5 variables:
# $ XCoord          : num  329746 329776 329806 329836 329866 ...
# $ YCoord          : num  1610848 1610848 1610848 1610848 1610848 ...
# $ pointid         : int  13182382 13182383 13182384 13182385 13182386 13182387 13182388 13182389 13182390 13182391 ...
# $ ground_ice      : int  NA NA NA NA NA NA NA NA NA NA ...
# $ all_thaw_feature: int  NA NA NA NA NA NA NA NA NA NA ...
#
# > table(ice_df$ground_ice)
# 
#     1      2      3      4      5      6      7 
# 323101 237763 284122  13396  10271   9937  34695 
# > table(ice_df$all_thaw_feature)
# 
#     1      2      3 
# 5658 109760   2135 

# Check if there are similarly weird small values in the coordinates
summary(ice_df[,c(1,2)])
# Seems all good.


# Merge the datasets
comb_df <- merge(veg_df, ice_df, by=c("XCoord", "YCoord"), all.x = TRUE, all.y = TRUE)

# rm(ice_df)

comb_df <- comb_df %>% rename(ground_ice_key = ground_ice) %>%
  mutate(all_thaw_feature = ifelse(all_thaw_feature==1, "thaw pond",
                                   ifelse(all_thaw_feature==2, "ice wedge",
                                          ifelse(all_thaw_feature==3, "thermokarst", NA)))) %>% 
  mutate(all_thaw_feature = as.factor(all_thaw_feature))

ice_key_df <- read.csv("ground_ice_key.csv", header = TRUE, na.strings = "NULL")

# > ice_key_df
# ground_ice_key   ground_ice_value ground_ice_likert is_water_body
# 1              1  "low to moderate"                 3             0
# 2              2    "absent to low"                 1             0
# 3              3 "moderate to high"                 4             0
# 4              4             "high"                 5             0
# 5              5             "none"                 0             0
# 6              6              "low"                 2             0
# 7              7       "water_body"                NA             1


comb_df <- comb_df %>% 
  merge(ice_key_df, by = "ground_ice_key", all.x = TRUE, all.y = FALSE) %>% 
  select(-ground_ice_key)


# fix the levels in ground_ice_value
levels(comb_df$ground_ice_value) <- c("absent to low", "high", "low to moderate", "low", 
                                        "moderate to high", "none", "water_body")

comb_df$ground_ice_value <- factor(comb_df$ground_ice_value, 
                                     levels = c("water_body", "none", "absent to low", "low", 
                                                "low to moderate", "moderate to high", "high"))
# > str(comb_df)
# 'data.frame':	13307480 obs. of  11 variables:
# $ XCoord           : num  311386 308086 295126 309046 313666 ...
# $ YCoord           : num  1668208 1666078 1668448 1664698 1659418 ...
# $ FuelTypeCl       : Factor w/ 24 levels "","Aquatic","Barren ground/gravel&dirt",..: 5 NA 3 16 20 NA NA NA 21 NA ...
# $ FAI_VegPlus1984  : int  16 NA 10 4 14 NA NA NA 14 NA ...
# $ FAI_VegPlus1994  : int  16 NA 14 4 14 NA NA NA 14 NA ...
# $ FAI_VegPlus2004  : int  16 NA 14 4 14 NA NA NA 8 NA ...
# $ FAI_VegPlus2014  : int  16 NA 14 4 14 NA NA NA 6 NA ...
# $ all_thaw_feature : Factor w/ 3 levels "ice wedge","thaw pond",..: NA NA NA NA NA NA NA NA NA NA ...
# $ ground_ice_value : Factor w/ 7 levels "water_body","none",..: 5 5 5 5 5 5 5 5 5 5 ...
# $ ground_ice_likert: int  3 3 3 3 3 3 3 3 3 3 ...
# $ is_water_body    : int  0 0 0 0 0 0 0 0 0 0 ...


rm(ice_df, ice_key_df, sr_ice_df, veg_df)

# #................................................................................
# # adjusted vegetation data
# veg_df <- read.csv("permafrost_data3.txt", header = TRUE, na.strings = "NULL")
# 
# str(veg_df)
# summary(veg_df)
# table(veg_df$ABOVE_1984_adjusted)
# 
# 
# # I won't bother proceeding with this one until Monika and Dmitry figure out what the
# # keys stand for.



# Vegetation data
# From my extrapolation for "Black Spruce"
load("C:/Users/Anna/Documents/NNA-AURA/Fairbanks vegetation/Extrapolation models/4-24 Decadal Fairbanks BS in Evergreen/Decadal ABoVE veg plus.RData")



###############################################################
# Visual checks on the variables for ground ice in the ground_ice source region


# the sourse region is where we have training data for ground_ice
sr_df <- comb_df %>% filter(!is.na(ground_ice_value))
summary(sr_df)

# Look at all_thaw_feature
plt <- sr_df  %>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_tile(aes(fill=all_thaw_feature))


ggsave(plt, file="./Graphs/SR_ice all_thaw_feature.png", height = 4, width = 7)



# Look at ground_ice_likert
plt <- sr_df  %>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_tile(aes(fill=ground_ice_likert))


ggsave(plt, file="./Graphs/SR_ice ground_ice_likert.png", height = 4, width = 7)


# Look at ground_ice_value
plt <- sr_df  %>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_tile(aes(fill=ground_ice_value)) +
  scale_fill_manual(values 
                    = c("water_body" = "blue",
                        "none" = "black",
                        "absent to low" = "gray",
                        "low" = "green",
                        "low to moderate" = "yellow",
                        "moderate to high" = "orange", 
                        "high" = "red"))


ggsave(plt, file="./Graphs/SR_ice ground_ice_value.png", height = 4, width = 7)

rm(plt)

##############################################################

# Combining multiple datasets that I have from vegetation by X and Y coordinate.
# Since some of the datasets have coordinates a little off,
# I will first round the coordinates to one digit.

# Loads dataframe "df" that contains variables for FNSB CWPP region
load("C:/Users/Anna/Documents/NNA-AURA/Fairbanks vegetation/FNSB CWPP region data/Fairbanks CWPP combined df 3-16.RData")

# > str(df)
# 'data.frame':	13307500 obs. of  36 variables:
#   $ XCoord                 : num  249736 249736 249736 249736 249736 ...
# $ YCoord                 : num  1704328 1704358 1704388 1704418 1704448 ...
# $ NewVeg7                : int  5 5 3 3 3 5 5 5 3 3 ...
# $ ELEV                   : num  290 293 296 301 305 ...
# $ ASPECT                 : num  NA NA NA NA NA NA NA NA NA NA ...
# $ SLOPE                  : num  NA NA NA NA NA NA NA NA NA NA ...
# $ GS_TEMP_FNSB_1996TO2005: num  13.5 13.5 13.5 13.5 13.5 ...
# $ S_TEMP_FNSB_1996TO2005 : num  15 15 15 15 15 ...
# $ W_TEMP_FNSB_1996TO2005 : num  -16.2 -16.2 -16.2 -16.2 -16.2 ...
# $ GS_PPT_FNSB_1996TO2005 : num  13.5 13.5 13.5 13.5 13.5 ...
# $ GS_TEMP_FNSB_2005      : num  14.6 14.6 14.6 14.6 14.6 ...
# $ S_TEMP_FNSB_2005       : num  15.6 15.6 15.6 15.6 15.6 ...
# $ W_TEMP_FNSB_2005       : num  -17.8 -17.8 -17.8 -17.8 -17.8 ...
# $ GS_PPT_FNSB_2005       : num  209 209 209 209 209 ...
# $ CANOPYHT               : num  100 100 0 0 0 49 100 100 100 100 ...
# $ CANOPY                 : num  NA NA NA NA NA NA NA NA NA NA ...
# $ BURNP                  : num  NA NA NA NA NA NA NA NA NA NA ...
# $ CWPP_VEG               : num  NA NA NA NA NA NA NA NA NA NA ...
# $ newVeg4_urban          : int  5 5 3 3 3 5 5 5 6 6 ...
# $ isWetland              : Factor w/ 2 levels "0","1": NA NA NA NA NA NA NA NA NA NA ...
# $ Canopy_estimate        : num  -5.8 -5.8 -2.53 -2.53 -2.53 ...
# $ NALCMS                 : int  6 6 5 5 5 1 1 1 2 5 ...
# $ FORESTVEG              : int  26 26 26 26 26 9 9 9 7 7 ...
# $ Fire2005               : num  -9999 -9999 -9999 -9999 -9999 ...
# $ Aspect_North           : num  NA NA NA NA NA NA NA NA NA NA ...
# $ Aspect_East            : num  NA NA NA NA NA NA NA NA NA NA ...
# $ MUKEY                  : Factor w/ 380 levels "1712802","2025489",..: 240 240 240 240 240 250 250 250 240 240 ...
# $ Veg_category           : Factor w/ 11 levels "Bare Ground",..: 8 8 5 5 5 8 8 8 5 5 ...
# $ isUrban                : num  0 0 0 0 0 0 0 0 0 0 ...
# $ AWS                    : num  NA NA 8.33 NA NA 8.33 NA 8.33 NA 8.33 ...
# $ DRAINAGE               : Factor w/ 8 levels " ","Very poorly drained",..: NA NA 6 NA NA 6 NA 6 NA 6 ...
# $ PHWATER                : num  NA NA 5.9 NA NA 5.9 NA 5.9 NA 5.9 ...
# $ PHSURFACE              : num  NA NA 5.2 NA NA 5.2 NA 5.2 NA 5.2 ...
# $ SOILTAXONOMY           : Factor w/ 48 levels " ","Aquic Cryofluvents",..: NA NA 37 NA NA 37 NA 37 NA 37 ...
# $ WETLAND_NWIGEN         : Factor w/ 9 levels "1","2","3","4",..: 9 9 9 9 9 9 9 9 9 9 ...
# $ ABoVE_veg              : Factor w/ 15 levels "Barren","Bog",..: 4 4 8 3 3 4 4 4 4 4 ...

# Only some of these variables are relevant (or of good-enough quality) for our purposes here.

names(df)

df <- df %>% select(
  XCoord, YCoord, # Coordinates
  ELEV, SLOPE, Aspect_North, Aspect_East,   # Topography
  S_TEMP_FNSB_1996TO2005, W_TEMP_FNSB_1996TO2005, S_TEMP_FNSB_2005, W_TEMP_FNSB_2005, # Temperatures
  isWetland, WETLAND_NWIGEN, # Wetland status
  isUrban, NALCMS, ABoVE_veg # Vegetation info (ABoVe veg is from 2005)
)

# Loads dataframe "fire_df" that contains a more comprehensive record of fire history
load("C:/Users/Anna/Documents/NNA-AURA/Fairbanks vegetation/FNSB CWPP region data/fire_df 1-31.RData")

# I will take the latest fire record from 2018, and include it for analysis
fire_df <- fire_df %>% select(XCoord, YCoord, F2018) %>% rename(Fire2018 = F2018)



# Merge the datasets
df <- merge(df, fire_df, by=c("XCoord", "YCoord"), all.x = TRUE, all.y = FALSE)



comb_df <- comb_df %>%
  mutate(XCoord = round(XCoord, digits = 1)) %>%
  mutate(YCoord = round(YCoord, digits = 1))


# Merge the datasets
comb_df <- merge(comb_df, df, by=c("XCoord", "YCoord"), all.x = TRUE, all.y = FALSE)

rm(df, fire_df)


# NWIGEN wetlands key
wetlands_df <- read.csv("Wetlands_NWIGen_key.csv", header = TRUE, na.strings = "NULL")

# > wetlands_df
#   NWIGEN_key                      NWIGEN_value
# 1          1                            Upland
# 2          2                              Lake
# 3          3 Freshwater Forested/Shrub Wetland
# 4          4                          Riverine
# 5          5                   Freshwater Pond
# 6          6       Freshwater Emergent Wetland
# 7          7              Freshwater Bryophyte
# 8          8    Estuarine and Marine Deepwater


comb_df <- comb_df %>% 
  rename(NWIGEN_key = WETLAND_NWIGEN) %>%
  merge(wetlands_df, by = "NWIGEN_key", all.x = TRUE, all.y = FALSE) %>%
  select(-NWIGEN_key)

rm(wetlands_df)

#####################################################################


# Minor edits
comb_df <- comb_df %>%
  rename(ABoVE_veg2005 = ABoVE_veg)


# the sourse region is where we have training data for ground_ice
sr_df <- comb_df %>% filter(!is.na(ground_ice_value))

save(comb_df,  file = "permafrost_df FNSB CWPP.RData")

save(sr_df,  file = "permafrost_df source.RData")


