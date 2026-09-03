import SwiftUI

@main
struct CoreYOLODemoApp: App {
    var body: some Scene {
        WindowGroup {
            CameraDetectView()
                .statusBarHidden(true)
        }
    }
}
