cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})


project(untitled)
include_directories(/home/telly/ncs/v2.8.0/emlearn/emlearn)


target_sources(app PRIVATE src/main.c)
