#include <stdio.h>
#include <string.h>

void demonstrate_string_basics() {
    printf("\n=== String Basics ===\n");
    
    // C-style string is char array ending with '\0'
    char str[] = "Hello";
    printf("String: %s\n", str);
    printf("Length: %zu\n", strlen(str));  // 5 (not counting '\0')
    printf("Size: %zu\n", sizeof(str));    // 6 (includes '\0')
    
    // Individual characters
    printf("First char: %c (ASCII: %d)\n", str[0], str[0]);  // 'H' (72)
    printf("Last char before \\0: %c\n", str[4]);            // 'o'
    printf("Null terminator: %d\n", str[5]);                 // 0 ('\0')
}

void demonstrate_string_comparison() {
    printf("\n=== String Comparison ===\n");
    
    const char* str1 = "Hello";
    const char* str2 = "Hello";
    const char* str3 = "World";
    
    printf("strcmp(Hello, Hello): %d\n", strcmp(str1, str2));  // 0 (equal)
    printf("strcmp(Hello, World): %d\n", strcmp(str1, str3));  // < 0 (str1 < str3)
    printf("strcmp(World, Hello): %d\n", strcmp(str3, str1));  // > 0 (str3 > str1)
}

void demonstrate_string_copying() {
    printf("\n=== String Copying ===\n");
    
    char src[] = "Source String";
    char dest[20];  // Make sure buffer is large enough
    
    // Unsafe copy (don't use in real code)
    strcpy(dest, src);
    printf("strcpy result: %s\n", dest);
    
    // Safe copy with size limit
    char small_dest[5];
    strncpy(small_dest, src, sizeof(small_dest) - 1);
    small_dest[sizeof(small_dest) - 1] = '\0';  // Ensure null termination
    printf("strncpy result (limited): %s\n", small_dest);
}

void demonstrate_string_formatting() {
    printf("\n=== String Formatting ===\n");
    
    char buffer[50];
    const char* name = "QWEN";
    int version = 3;
    float temperature = 0.6f;
    
    // sprintf (unsafe - no buffer size check)
    sprintf(buffer, "Model: %s-%d (temp=%.1f)", name, version, temperature);
    printf("sprintf result: %s\n", buffer);
    
    // snprintf (safe - specifies buffer size)
    snprintf(buffer, sizeof(buffer), "Model: %s-%d (temp=%.1f)", 
             name, version, temperature);
    printf("snprintf result: %s\n", buffer);
}

int main() {
    demonstrate_string_basics();
    demonstrate_string_comparison();
    demonstrate_string_copying();
    demonstrate_string_formatting();
    return 0;
}
