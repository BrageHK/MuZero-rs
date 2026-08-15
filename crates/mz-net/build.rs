fn main() {
    tonic_prost_build::compile_protos("proto/muzero.proto").expect("Failed to compile muzero.proto");
}
