fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .compile_protos(&["proto/quote.proto", "proto/market.proto","proto/test.proto"], &["proto"])?;
    Ok(())
}
