from torch import nn

class FSRCNN(nn.Module):
    def __init__(self,
                scale_factor, 
                num_channels=1,
                # number of filters in the first and last layers
                d=56,
                # number of filters in the shrinking and mapping layers
                s=12,
                # number of mapping layers
                m=4):

        super(FSRCNN, self).__init__()

        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 
                      d, 
                      kernel_size=5, 
                      padding=5//2),
            nn.PReLU(d)
        )

        self.mid_part = [
            nn.Conv2d(d, s, kernel_size=1), 
            nn.PReLU(s)
        ]

        # For every mapping layer
        for _ in range(m):
            # Add a convolution layer followed by a PReLU activation function
            self.mid_part.extend([
                nn.Conv2d(s, 
                          s, 
                          kernel_size=3, 
                          padding=3//2),
                nn.PReLU(s)])
        
        # Add one more convolution layer followed by a PReLU activation function
        self.mid_part.extend([
            nn.Conv2d(s, 
                      d, 
                      kernel_size=1),
            nn.PReLU(d)])
        
        # Convert the list of layers into a sequential module
        self.mid_part = nn.Sequential(*self.mid_part)

        
        self.last_part = nn.ConvTranspose2d(d, 
                                            num_channels, 
                                            kernel_size=9, 
                                            stride=scale_factor, 
                                            padding=9//2,
                                            output_padding=scale_factor-1)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x