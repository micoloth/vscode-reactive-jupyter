

use wasm_bindgen::prelude::*;

// Expose the function to JavaScript via wasm-bindgen
#[wasm_bindgen]
pub fn to_uppercase(input: &str) -> String {
    input.to_uppercase()
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    // Test the `to_uppercase` function:
    #[test]
    fn test_to_uppercase() {
        let input = "hello";
        let expected = "HELLO";
        let result = to_uppercase(input);
        assert_eq!(result, expected);
    }
}
