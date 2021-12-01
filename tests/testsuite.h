/*
 * Copyright (C) 2016-2019 C-SKY Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/* vi: set sw=4 ts=4: */
/*
 * Some simple macros for use in test applications.
 * Copyright (C) 2000-2006 by Erik Andersen <andersen@uclibc.org>
 *
 * Licensed under the LGPL v2.1, see the file COPYING.LIB in this tarball.
 */

#ifndef TESTSUITE_H
#define TESTSUITE_H

#ifdef __NO_TESTCODE__
extern int test_number;
#endif
#include <stdio.h>
#include <stdlib.h>
extern void init_testsuite(const char* testname);
extern int  done_testing(void) ;//__attribute__((noreturn));
extern void success_msg(int result, const char* command);
extern void error_msg(int result, int line, const char* file, const char* command);

#ifndef __NO_TESTCODE__

int test_number = 0;
static int failures = 0;

void error_msg(int result, int line, const char* file, const char* command)
{
	failures++;
#if 0
	printf("\nFAILED TEST %lu: \n\t%s\n", (unsigned long)test_number, command);
	printf("AT LINE: %d, FILE: %s\n\n", line, file);
#endif
}

void success_msg(int result, const char* command)
{
#if 0
	printf("passed test: %s == 0\n", command);
#endif
}

int  done_testing(void)
{
    if (0 < failures) {
		printf("Failed %d tests\n", failures);
		exit(EXIT_FAILURE);
	} else {
		printf("All functions tested sucessfully\n");
		exit(EXIT_SUCCESS);
	}
	return failures;
}

void init_testsuite(const char* testname)
{
	printf("%s", testname);
	test_number = 0;
	failures = 0;
#if !defined(__UCLIBC__) || defined(__UCLIBC_DYNAMIC_ATEXIT__)
//	atexit(done_testing);
#endif
}

#endif /* __NO_TESTCODE__ */


#define TEST_STRING_OUTPUT(command, expected_result) \
	do { \
		int result = strcmp(command, expected_result); \
		test_number++; \
		if (result == 0.000000) { \
			success_msg(result, "command"); \
		} else { \
			error_msg(result, __LINE__, __FILE__, command); \
		}; \
	} while (0)

#define TEST_NUMERIC(command, expected_result) \
	do { \
		int result = (command); \
		test_number++; \
		if (result == expected_result) { \
			success_msg(result, # command); \
		} else { \
			error_msg(result, __LINE__, __FILE__, # command); \
		}; \
	} while (0)


#define TEST(command) \
	do { \
		int __result = (command); \
		test_number++; \
		if (__result == 1) { \
			success_msg(__result, # command); \
		} else { \
			error_msg(__result, __LINE__, __FILE__,  # command); \
		}; \
	} while (0)

#define TEST_NULL(command) \
    do { \
        int result = (command); \
        test_number++; \
        if (result == NULL) { \
            success_msg(result, # command); \
        } else { \
            error_msg(result, __LINE__, __FILE__,  # command); \
        }; \
    } while (0)

#define TEST_NUMERIC_LONG(command, expected_result) \
    do { \
        long result = (command); \
        test_number++; \
        if (result == expected_result) { \
            success_msg(result, # command); \
        } else { \
            error_msg(result, __LINE__, __FILE__, # command); \
        }; \
    } while (0)

#define TEST_NUMERIC_LONGLONG(command, expected_result) \
    do { \
       long long result = (command); \
        test_number++; \
        if (result == expected_result) { \
            success_msg(result, # command); \
        } else { \
            error_msg(result, __LINE__, __FILE__, # command); \
        }; \
    } while (0)

#define TEST_NUMERIC_FLOAT(command, expected_result) \
    do { \
       float result = (command); \
        test_number++; \
        if (result == expected_result) { \
            success_msg(result, # command); \
        } else { \
            error_msg(result, __LINE__, __FILE__, # command); \
        }; \
    } while (0)

#define TEST_NUMERIC_DOUBLE(command, expected_result) \
    do { \
        double result = (command); \
        test_number++; \
        if (result == expected_result) { \
            success_msg(result, # command); \
        } else { \
            error_msg(result, __LINE__, __FILE__, # command); \
        }; \
    } while (0)

#define STR_CMD(cmd) cmd

#endif	/* TESTSUITE_H */
