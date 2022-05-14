set(IS_FALSE_PATTERN
    "^[Oo][Ff][Ff]$|^0$|^[Ff][Aa][Ll][Ss][Ee]$|^[Nn][Oo]$|^[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|.*-[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|^$"
)
set(IS_TRUE_PATTERN
    "^[Oo][Nn]$|^[1-9][0-9]*$|^[Tt][Rr][Uu][Ee]$|^[Yy][Ee][Ss]$|^[Yy]$")

macro(__brt_option variable description value)
    if(NOT DEFINED ${variable})
        set(${variable}
            ${value}
            CACHE STRING ${description})
    endif()
endmacro()

set(BRT_ALL_OPTIONS)

# ##############################################################################
# An option that the user can select. Can accept condition to control when
# option is available for user. Usage: brt_option(<option_variable> "doc string"
# <initial value or boolean expression> [IF <condition>])
macro(brt_option variable description value)
    set(__value ${value})
    set(__condition "")
    set(__varname "__value")
    list(APPEND BRT_ALL_OPTIONS ${variable})
    foreach(arg ${ARGN})
        if(arg STREQUAL "IF" OR arg STREQUAL "if")
            set(__varname "__condition")
        else()
            list(APPEND ${__varname} ${arg})
        endif()
    endforeach()
    unset(__varname)
    if("${__condition}" STREQUAL "")
        set(__condition 2 GREATER 1)
    endif()

    if(${__condition})
        if("${__value}" MATCHES ";")
            if(${__value})
                __brt_option(${variable} "${description}" ON)
            else()
                __brt_option(${variable} "${description}" OFF)
            endif()
        elseif(DEFINED ${__value})
            if(${__value})
                __brt_option(${variable} "${description}" ON)
            else()
                __brt_option(${variable} "${description}" OFF)
            endif()
        else()
            __brt_option(${variable} "${description}" "${__value}")
        endif()
    else()
        unset(${variable} CACHE)
    endif()
endmacro()

macro(brt_src_exclude SOURCE EXCLUDE)
foreach(TMP_SRC ${${SOURCE}})
    string(FIND ${TMP_SRC} ${EXCLUDE} EXCLUDE_DIR_FOUND)
    if(NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
        message(STATUS "removing ${TMP_SRC}")
        list(REMOVE_ITEM ${SOURCE} ${TMP_SRC})
    endif()
endforeach(TMP_SRC)
endmacro()
